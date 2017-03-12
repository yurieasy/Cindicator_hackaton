import pandas as pd
from elementwise import rmsle
import numpy as np

#  Back testing
def back_testing(data):
    tickers_dict_r_min = {}
    tickers_dict_r_max = {}
    tickers_dict_f_min = {}
    tickers_dict_f_max = {}
    tickers_dict_f_min_weighted = {}
    tickers_dict_f_max_weighted = {}
    ds = data.groupby('event_id')
    for i in ds:
        mean_max = i[1]['prediction_max'].values.mean()
        mean_min = i[1]['prediction_min'].values.mean()
        mean_max_weighted = np.average(i[1]['prediction_max'].values, weights=i[1]['max_weight'].values)
        mean_min_weighted = np.average(i[1]['prediction_min'].values, weights=i[1]['min_weight'].values)
        real_max = i[1]['real_max'].values.mean()
        real_min = i[1]['real_min'].values.mean()
        ticker = i[1]['ticker_id'][0]
        if ticker not in tickers_dict_r_min.keys():
            tickers_dict_r_min[ticker] = []
            tickers_dict_r_max[ticker] = []
            tickers_dict_f_min[ticker] = []
            tickers_dict_f_max[ticker] = []
            tickers_dict_f_min_weighted[ticker] = []
            tickers_dict_f_max_weighted[ticker] = []
        tickers_dict_r_min[ticker].append(real_min)
        tickers_dict_r_max[ticker].append(real_max)
        tickers_dict_f_min[ticker].append(mean_min)
        tickers_dict_f_max[ticker].append(mean_max)
        tickers_dict_f_min_weighted[ticker].append(mean_min_weighted)
        tickers_dict_f_max_weighted[ticker].append(mean_max_weighted)

    result_min = {}
    result_max = {}
    result_min_weighted = {}
    result_max_weighted = {}

    for ticker in tickers_dict_r_min.keys():
        rmsle_min = rmsle(tickers_dict_r_min[ticker], tickers_dict_f_min[ticker])
        rmsle_max = rmsle(tickers_dict_r_max[ticker], tickers_dict_f_max[ticker])
        rmsle_min_weighted = rmsle(tickers_dict_r_min[ticker], tickers_dict_f_min_weighted[ticker])
        rmsle_max_weighted = rmsle(tickers_dict_r_max[ticker], tickers_dict_f_max_weighted[ticker])
        result_min[ticker] = rmsle_min
        result_max[ticker] = rmsle_max
        result_min_weighted[ticker] = rmsle_min_weighted
        result_max_weighted[ticker] = rmsle_max_weighted

    #  RMSLE for popular ticker
    ticker = 'c5e2ca55-3606-40ad-aae5-55be180a7de5'
    rmsle_min_for_ticker = rmsle(tickers_dict_r_min[ticker], tickers_dict_f_min[ticker])
    rmsle_max_for_ticker = rmsle(tickers_dict_r_max[ticker], tickers_dict_f_max[ticker])
    rmsle_min_weighted_for_ticker = rmsle(tickers_dict_r_min[ticker], tickers_dict_f_min_weighted[ticker])
    rmsle_max_weighted_for_ticker = rmsle(tickers_dict_r_max[ticker], tickers_dict_f_max_weighted[ticker])

    #  RMSLE mean for all tickers
    res_min = np.mean(list(result_min.values()))
    res_max = np.mean(list(result_max.values()))
    res_min_weighted = np.mean(list(result_min_weighted.values()))
    res_max_weighted = np.mean(list(result_max_weighted.values()))

    #  Output
    print('Baseline for popular ticker:')
    print(str(rmsle_min_for_ticker) + ' - RMSLE min')
    print(str(rmsle_max_for_ticker) + ' - RMSLE max\n')
    print('Weighted for popular ticker:')
    print(str(rmsle_min_weighted_for_ticker) + ' - RMSLE min')
    print(str(rmsle_max_weighted_for_ticker) + ' - RMSLE max\n')
    print('Baseline for all tickers')
    print(str(res_min) + ' - RMSLE min')
    print(str(res_max) + ' - RMSLE max\n')
    print('Weighted for all tickers:')
    print(str(res_min_weighted) + ' - RMSLE min')
    print(str(res_max_weighted) + ' - RMSLE max')


#  RUN!!
df = pd.DataFrame.from_csv('data_res.csv')
back_testing(df)
