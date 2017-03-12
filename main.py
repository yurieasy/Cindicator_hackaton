import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from cleaning import remove_outliers
from generate_weights import generate_weights
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
    weights = generate_weights(df)
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


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ticker = "c5e2ca55-3606-40ad-aae5-55be180a7de5"

    df = pd.DataFrame.from_csv('pricing_answers.csv', header=0, index_col=None, parse_dates=['event_finished_at'])
    df = df.loc[df.ticker_id == ticker]

    df = remove_outliers(df)
    results = predict(df)

    metric = mean_absolute_error

    base_min = metric(results["y_true_min"], results["y_baseline_min"])
    weight_min = metric(results["y_true_min"], results["y_weighted_min"])
    base_max = metric(results["y_true_max"], results["y_baseline_max"])
    weight_max = metric(results["y_true_max"], results["y_weighted_max"])
    print("Min (baseline/weighted)")
    print(base_min)
    print(weight_min)
    print("Advantage: %f" % (base_min - weight_min))
    print("Max (baseline/weighted)")
    print(base_max)
    print(weight_max)
    print("Advantage: %f" % (base_max - weight_max))

    plot_ticker(results)
    logging.info(mean_absolute_error(results["y_true"], results["y_baseline"]))
    logging.info(mean_absolute_error(results["y_true"], results["y_weighted"]))
    logging.info(mean_absolute_error(results["y_true"], results["y_adjusted"]))
