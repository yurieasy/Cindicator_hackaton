import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir = "C:\\Workspace\\AiHackathon\\"
ds = pd.DataFrame.from_csv(dir + 'pricing_answers.csv', header=0, index_col=None, parse_dates=['event_finished_at'])

# Normalisation
G = ds.groupby("ticker_id")
MIN_FOR_TICKER = pd.concat([G["real_min"].min(), G["prediction_min"].min()], axis=1).min(axis=1)
MAX_FOR_TICKER = pd.concat([G["real_max"].max(), G["prediction_max"].max()], axis=1).max(axis=1)
ds["MIN_FOR_TICKER"] = ds["ticker_id"]
ds["MAX_FOR_TICKER"] = ds["ticker_id"]
ds = ds.replace({"MIN_FOR_TICKER": MIN_FOR_TICKER})
ds = ds.replace({"MAX_FOR_TICKER": MAX_FOR_TICKER})

def mne_norm(field, field_max, field_min):
    ds[field] = (ds[field] - ds[field_min]) / (ds[field_max] - ds[field_min])

mne_norm("real_min", "MAX_FOR_TICKER", "MIN_FOR_TICKER")
mne_norm("prediction_min", "MAX_FOR_TICKER", "MIN_FOR_TICKER")
mne_norm("real_max", "MAX_FOR_TICKER", "MIN_FOR_TICKER")
mne_norm("prediction_max", "MAX_FOR_TICKER", "MIN_FOR_TICKER")

ds["prediction_width"] = ds["prediction_max"] - ds["prediction_min"]
ds["real_width"] = ds["real_max"] - ds["real_min"]
ds["prediction_center"] = (ds["prediction_max"] + ds["prediction_min"]) / 2
ds["real_center"] = (ds["real_max"] + ds["real_min"]) / 2
ds["error_center"] = ds["prediction_center"] - ds["real_center"]
ds["mae_center"] = abs(ds["prediction_center"] - ds["real_center"])
ds["error_width"] = ds["prediction_width"] - ds["real_width"]

ticker = "4c0469d0-2569-46ae-902b-c7fbd4ae0cb1"
ticker = "c5e2ca55-3606-40ad-aae5-55be180a7de5"
ds1t = ds[ds["ticker_id"] == ticker]
dates = set(ds1t["event_finished_at"].values)

user_mem = {}
for uid in ds.user_id.values:
    user_mem[uid] = []

def get_user_params(uid):
    s = None
    m = None
    if len(user_mem[uid])==0:
        s = 1.0
        m = 0.0
    else:
        s = np.std(user_mem[uid])
        m = np.mean(user_mem[uid])
    return m, s

def update_user(uid, error):
    user_mem[uid].append(error)

fout = open("res.csv", "wt")

fout.write("cur_date,uid,real_center,prediction_center,corrected_prediction_center,sigma0,usermem\n")

i = 0
for cur_date in dates:
    print("Date", cur_date)
    ds1tdt = ds1t[ds1t["event_finished_at"] == cur_date]
    uids = set(ds1tdt["user_id"].values)
    for uid in uids:
        print("User", uid)
        userguess = ds1tdt[ds1tdt["user_id"] == uid]
        mu0, sigma0 = get_user_params(uid)

        error = (userguess["prediction_max"] - userguess["prediction_min"] - userguess["real_max"] + userguess["real_min"]).values[0]
        update_user(uid, error)

        fout.write("{0},{1},{2},{3},{4},{5},{6}\n".format(
            cur_date,
            uid,
            userguess["real_center"].values[0],
            userguess["prediction_center"].values[0],
            userguess["prediction_center"].values[0] - mu0,
            sigma0,
            len(user_mem[uid])-1
        ))

fout.close()