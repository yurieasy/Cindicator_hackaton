import numpy as np
import pandas as pd


def linex(err, alpha):
    return np.exp(-alpha * err) + alpha * err - 1


def get_user_params(user_mem, uid):
    if len(user_mem[uid]) < 2:
        s = 1000000.0
        m = 0.0
    else:
        s = np.std(user_mem[uid])
        m = np.mean(user_mem[uid])
    return m, s


def generate_weights(df):
    dates = df["event_finished_at"].unique()
    user_mem_min = {}
    user_mem_max = {}
    for uid in df.user_id.values:
        user_mem_min[uid] = []
        user_mem_max[uid] = []

    columns = ["event_finished_at", "user_id", "mu_min", "sigma_min", "mu_max", "sigma_max"]
    result = []

    for cur_date in sorted(dates):
        IS_LINEX = True
        ds1tdt = df[df["event_finished_at"] == cur_date]
        uids = set(ds1tdt["user_id"].values)
        for uid in uids:
            userguess = ds1tdt[ds1tdt["user_id"] == uid]
            mu_min, sigma_min = get_user_params(user_mem_min, uid)
            mu_max, sigma_max = get_user_params(user_mem_max, uid)
            if IS_LINEX:
                alpha = 0.02
                max_e = userguess["prediction_max"] - userguess["real_max"]
                min_e = userguess["prediction_min"] - userguess["real_min"]
                error_max = linex(max_e, -alpha)
                error_min = linex(min_e, -alpha)
            else:
                error_min = (userguess["real_min"] - userguess["prediction_min"])
                error_max = (userguess["real_max"] - userguess["prediction_max"])
            user_mem_min[uid].append(error_min.values[0])
            user_mem_max[uid].append(error_max.values[0])

            result.append([cur_date, uid, mu_min, sigma_min, mu_max, sigma_max])

    return pd.DataFrame(result, columns=columns)
