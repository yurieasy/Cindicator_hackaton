# import numpy
# import pandas as pd
#
# from metrics import rmsle
#
# tickers_dict_r_min = {}
# tickers_dict_r_max = {}
# tickers_dict_f_min = {}
# tickers_dict_f_max = {}
#
# ds = pd.DataFrame.from_csv('pricing_answers.csv').groupby('event_id')
# for i in ds:
#     mean_max = i[1]['prediction_max'].values.mean()
#     mean_min = i[1]['prediction_min'].values.mean()
#     real_max = i[1]['real_max'].values.mean()
#     real_min = i[1]['real_min'].values.mean()
#     ticker = i[1]['ticker_id'][0]
#     if ticker not in tickers_dict_r_min.keys():
#         tickers_dict_r_min[ticker] = []
#     if ticker not in tickers_dict_r_max.keys():
#         tickers_dict_r_max[ticker] = []
#     if ticker not in tickers_dict_f_min.keys():
#         tickers_dict_f_min[ticker] = []
#     if ticker not in tickers_dict_f_max.keys():
#         tickers_dict_f_max[ticker] = []
#     tickers_dict_r_min[ticker].append(real_min)
#     tickers_dict_r_max[ticker].append(real_max)
#     tickers_dict_f_min[ticker].append(mean_min)
#     tickers_dict_f_max[ticker].append(mean_max)
#
# result_min = {}
# result_max = {}
#
# for ticker in tickers_dict_r_min.keys():
#     rmsle_min = rmsle(tickers_dict_r_min[ticker], tickers_dict_f_min[ticker])
#     rmsle_max = rmsle(tickers_dict_r_max[ticker], tickers_dict_f_max[ticker])
#     result_min[ticker] = rmsle_min
#     result_max[ticker] = rmsle_max
#
# res_min = numpy.mean(list(result_min.values()))
# res_max = numpy.mean(list(result_max.values()))
# print(res_min)
# print(res_max)
