import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt

dir = "C:\\Workspace\\AiHackathon\\"
ds = pd.DataFrame.from_csv(dir + 'res.csv', header=0, index_col=None, parse_dates=['cur_date'])

#fout.write("cur_date,uid,real_center,prediction_center,corrected_prediction_center,sigma0\n")

dates = set(ds["cur_date"].values)

m1s = []
m2s = []

for date in dates:
    dsdt = ds[ds["cur_date"] == date]
    #uids = set(dsdt["user_id"].values)
    D = dsdt.to_dict(orient='records')
    N = len(D)

    real = D[0]["real_center"]

    # baseline
    S = 0
    for d in D:
        S += d["prediction_center"]
    S /= N
    M1 = abs(S - real)

    # stat model
    WS = 0.0

    Weights = []
    for d in D:
        SIG = d["sigma0"]

        if(d["usermem"] < 2):
            SIG = 1000.0

        Eps = 0.000001

        if(SIG < Eps):
            SIG = Eps

        W = 1.0 / sqrt(SIG)
        Weights.append(W)
    WS = sum(Weights)

    S = 0.0
    for i in range(N):
        S += (Weights[i] / WS) * D[i]["corrected_prediction_center"]
    M2 = abs(S - real)

    m1s.append(M1)
    m2s.append(M2)

    print("{0}, {4}, {1}, {2}, {3}".format(date, M1, M2, 100*(M1-M2)/M1, N))

print(np.mean(m1s[2:]), np.mean(m2s[2:]))