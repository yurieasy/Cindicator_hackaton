import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from generate_weights import generate_weights


def adjust_weights(weights):
    weights = 1 / weights.apply(np.sqrt)
    return weights.div(weights.sum(axis=0), axis=0)


if __name__ == "__main__":
    ticker = "c5e2ca55-3606-40ad-aae5-55be180a7de5"

    df = pd.DataFrame.from_csv('pricing_answers.csv', header=0, index_col=None, parse_dates=['event_finished_at'])
    df = df.loc[df.ticker_id == ticker]

    if not os.path.isfile("weights.csv"):
        weights = generate_weights(df)
        weights.to_csv("weights.csv", index=None)
    else:
        weights = pd.read_csv("weights.csv")

    dates = set(weights["event_finished_at"].values)

    y_true = []
    baseline = []
    weighted = []
    adjusted = []

    for date in sorted(dates):
        day_weights = weights[weights["event_finished_at"] == date]

        day_weights["sigma_min"] = adjust_weights(day_weights["sigma_min"])
        day_weights["sigma_max"] = adjust_weights(day_weights["sigma_max"])

        answers = df[df["event_finished_at"] == date]
        day_weights = day_weights.set_index("user_id")
        answers = answers.set_index("user_id")
        merged = pd.concat([answers, day_weights], axis=1)

        real = merged["real_min"].values[0]

        predicted_baseline = merged["prediction_min"].mean()
        predicted_weighted = (merged["prediction_min"] * merged["sigma_min"]).sum()
        predicted_adjust = ((merged["prediction_min"] - merged["mu_min"]) * merged["sigma_min"]).sum()

        y_true.append(real)
        baseline.append(predicted_baseline)
        weighted.append(predicted_weighted)
        adjusted.append(predicted_adjust)

        print("{0}, {1}, {2}, {3}".format(date, predicted_baseline, predicted_weighted, predicted_adjust))

    print(mean_absolute_error(y_true, baseline))
    print(mean_absolute_error(y_true, weighted))
    print(mean_absolute_error(y_true, adjusted))
