from matplotlib import pyplot as plt
import numpy as np


def plot_ticker(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    x_axis = np.array(list(range(len(df["event_finished_at"]))))

    ax1.plot(x_axis, df["y_baseline_min"], label='Baseline', drawstyle='steps')
    ax1.plot(x_axis, df["y_true_min"], label='True', drawstyle='steps')
    ax1.plot(x_axis, df["y_weighted_min"], label='Weighted', drawstyle='steps')
    ax1.legend(loc="best")

    ax2.plot(x_axis, df["y_baseline_max"], label='Baseline', drawstyle='steps')
    ax2.plot(x_axis, df["y_true_max"], label='True', drawstyle='steps')
    ax2.plot(x_axis, df["y_weighted_max"], label='Weighted', drawstyle='steps')
    ax2.legend(loc="best")

    width = 0.35

    ax3.bar(x_axis, abs(df["y_baseline_min"]-df["y_true_min"]), width, label="Baseline error")
    ax3.bar(x_axis+width, abs(df["y_weighted_min"]-df["y_true_min"]), width, color="green", label="Weighted error")
    ax3.legend(loc="best")

    ax4.bar(x_axis, abs(df["y_baseline_max"]-df["y_true_max"]), width, label="Baseline error")
    ax4.bar(x_axis+width, abs(df["y_weighted_max"]-df["y_true_max"]), width, color="green", label="Weighted error")
    ax4.legend(loc="best")

    fig.autofmt_xdate()

    plt.show()
