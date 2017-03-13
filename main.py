import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from cleaning import remove_outliers

from generate_weights import generate_weights
from generate_weights import linex

from generate_weights import LINEX_ALPHA

from visualise import plot_ticker
from metrics import rmsle, rmse

pd.options.mode.chained_assignment = None

def adjust_weights(weights):
    weights = 1 / weights.apply(np.sqrt)
    return weights.div(weights.sum(axis=0), axis=0)

def predict_sides(df):
    date = df["event_finished_at"].values[0]

    y_true_min = df["real_min"].values[0]
    predicted_baseline_min = df["prediction_min"].mean()
    predicted_weighted_min = (df["prediction_min"] * df["sigma_min"]).sum()
    predicted_adjust_min = ((df["prediction_min"] - df["mu_min"]) * df["sigma_min"]).sum()

    y_true_max = df["real_max"].values[0]
    predicted_baseline_max = df["prediction_max"].mean()
    predicted_weighted_max = (df["prediction_max"] * df["sigma_max"]).sum()
    predicted_adjust_max = ((df["prediction_max"] - df["mu_max"]) * df["sigma_max"]).sum()

    return [date, y_true_min, predicted_baseline_min, predicted_weighted_min, y_true_max,
            predicted_baseline_max, predicted_weighted_max]

def predict(df):

    # Weights generation
    # Flag switches on/of the linex loss function
    weights = generate_weights(df, False)

    # Writing weights to csv
    # weights.to_csv("weights.csv", index=None)
    # else:
    #     weights = pd.read_csv("weights.csv")

    dates = set(weights["event_finished_at"].values)

    results = []

    header = ["event_finished_at", "y_true_min", "y_baseline_min", "y_weighted_min",
              "y_true_max", "y_baseline_max", "y_weighted_max"]

    for date in sorted(dates):
        day_weights = weights[weights["event_finished_at"] == date]

        day_weights.loc[:, "sigma_min"] = adjust_weights(day_weights.loc[:, "sigma_min"])
        day_weights.loc[:, "sigma_max"] = adjust_weights(day_weights.loc[:, "sigma_max"])

        answers = df[df["event_finished_at"] == date]
        day_weights = day_weights.set_index("user_id")
        answers = answers.set_index("user_id")
        merged = pd.concat([answers, day_weights], axis=1)

        results.append(predict_sides(merged))

    return pd.DataFrame(results, columns=header)


def calc_metric_advantage(results, f_metric, name):
    base_min = f_metric(results["y_true_min"], results["y_baseline_min"])
    weight_min = f_metric(results["y_true_min"], results["y_weighted_min"])
    base_max = f_metric(results["y_true_max"], results["y_baseline_max"])
    weight_max = f_metric(results["y_true_max"], results["y_weighted_max"])
    print("Min %s (baseline/weighted)" % name)
    print(base_min)
    print(weight_min)
    print("Advantage: %f" % (base_min - weight_min))
    print("Max (baseline/weighted)")
    print(base_max)
    print(weight_max)
    print("Advantage: %f" % (base_max - weight_max))

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ticker = "c5e2ca55-3606-40ad-aae5-55be180a7de5"
    #ticker = "a0c756e7-481a-4a9e-bed1-32db7cd40279"

    df = pd.DataFrame.from_csv('pricing_answers.csv', header=0, index_col=None, parse_dates=['event_finished_at'])
    df = df.loc[df.ticker_id == ticker]

    results_dirty = predict(df)
    calc_metric_advantage(results_dirty, mean_absolute_error, "MAE")

    df = remove_outliers(df)
    result_clear = predict(df)

    results = result_clear
    results["y_baseline_min"] = results_dirty["y_baseline_min"]
    results["y_baseline_max"] = results_dirty["y_baseline_max"]

    #-------------------------------------------------------------------
    # Save data to csv for bot
    #-------------------------------------------------------------------
    def to_str(ts):
        ts = ts[0]
        ts = pd.to_datetime(str(ts))
        return ts.strftime('%Y-%m-%d')

    to_csv = results[['event_finished_at', 'y_weighted_min', 'y_weighted_max']].copy()
    to_csv['event_finished_at'] = to_csv['event_finished_at'].apply(to_str)
    to_csv.columns = ['end_time', 'averageWeightedMin', 'averageWeightedMax']
    to_csv.to_csv('weighted.csv')

    #-------------------------------------------------------------------
    # Apply metrics
    #-------------------------------------------------------------------
    calc_metric_advantage(results, rmsle, "RMSLE")
    calc_metric_advantage(results, mean_absolute_error, "MAE")

    def linex_func(vtrue, vpred):
        return np.mean(linex(vpred - vtrue, LINEX_ALPHA))

    calc_metric_advantage(results, linex_func, "LINEX")

    #-------------------------------------------------------------------
    # Draw plots
    #-------------------------------------------------------------------
    plot_ticker(results)
