# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 구간에서 train된 모델들에 대한 앙상블들을 생성하여 각각의 수익률과 함께 저장

import ensemble_ensemble_test as ep
import datetime
import pandas as pd
import numpy as np
import sys
import random

ep.selected_num = 3
ep.weights = [1 for i in range(ep.selected_num)]


last_trains = [
    '2021-12-31',
    '2022-01-15', '2022-01-31', '2022-02-15', '2022-02-28',
    '2022-03-15', '2022-03-31', '2022-04-15', '2022-04-30',
    '2022-05-15', '2022-05-31', '2022-06-15', '2022-06-30',
    '2022-07-15', '2022-07-31', '2022-08-15', '2022-08-31',
    '2022-09-15', '2022-09-30', '2022-10-15', '2022-10-31',
    '2022-11-15', '2022-11-30', '2022-12-15',
]
#    '2022-12-31',
#    '2023-01-15', '2023-01-31', '2023-02-15', '2023-02-28',
#]


start_times = [
    '2022/01/01/09:00', '2022/01/16/09:00', '2022/02/01/09:00', '2022/02/16/09:00',
    '2022/03/01/09:00', '2022/03/16/09:00', '2022/04/01/09:00', '2022/04/16/09:00',
    '2022/05/01/09:00', '2022/05/16/09:00', '2022/06/01/09:00', '2022/06/16/09:00',
    '2022/07/01/09:00', '2022/07/16/09:00', '2022/08/01/09:00', '2022/08/16/09:00',
    '2022/09/01/09:00', '2022/09/16/09:00', '2022/10/01/09:00', '2022/10/16/09:00',
    '2022/11/01/09:00', '2022/11/16/09:00', '2022/12/01/09:00', '2022/12/16/09:00',
]
#    '2023/01/01/09:00', '2023/01/16/09:00', '2023/02/01/09:00', '2023/02/16/09:00',
#    '2023/03/01/09:00',
#]

end_times = [
    '2022/01/15/15:00', '2022/01/31/15:00', '2022/02/15/15:00', '2022/02/28/15:00',
    '2022/03/15/15:00', '2022/03/31/15:00', '2022/04/15/15:00', '2022/04/30/15:00',
    '2022/05/15/15:00', '2022/05/31/15:00', '2022/06/15/15:00', '2022/06/30/15:00',
    '2022/07/15/15:00', '2022/07/31/15:00', '2022/08/15/15:00', '2022/08/31/15:00',
    '2022/09/15/15:00', '2022/09/30/15:00', '2022/10/15/15:00', '2022/10/31/15:00',
    '2022/11/15/15:00', '2022/11/30/15:00', '2022/12/15/15:00', '2022/12/31/15:00',
]

#    '2023/01/15/15:00', '2023/01/31/15:00', '2023/02/15/15:00', '2023/02/28/15:00',
#    '2023/03/15/15:00',
#]


ep.trading_9h = False

term = "ensemble(all120)_2022-01-01~2022-12-31"
ep.eval_arr = pd.read_csv("all_ensemble_results_2022-04-01~2023-04-06_5_0.5_5.csv", encoding='euc-kr').values[:, :5]
ep.reinfo_th = 0.5
ep.pred_term = 5


import os
folder = "best_ensemble_"+term + "_" + str(ep.reinfo_th) + "_" + str(ep.pred_term)
if not os.path.isdir(folder):
    os.makedirs(folder)

result_path = "all_ensemble_results_" + term + "_" + str(ep.reinfo_th) + "_" + str(ep.pred_term) + ".csv"
total_ensemble_path = "total_ensembles_" + term + "_" + str(ep.reinfo_th) + "_" + str(ep.pred_term) + ".csv"

def main():

    ensemble_models = ""
    for i in range(ep.selected_num):
        ensemble_models = ensemble_models + str(ep.selected_ensemble_types[i]) + "_"

    if ep.trading_9h:
        path = "9시시가에거래_ensemble_" + str(ep.pred_term) + "_reinfo_" + str(ep.reinfo_th) + "_" + str(ep.model_reinfo_th) + "_" + ensemble_models + ".xlsx"
    else:
        path = "9시거래없음_ensemble_" + str(ep.pred_term) + "_reinfo_" + str(ep.reinfo_th) + "_" + str(ep.model_reinfo_th) + "_" + ensemble_models + ".xlsx"

    path = folder + "/" + path

    length = len(last_trains)

    # 모델별 각 구간별 수익률 계산, 저장
    profit_sum = []
    profit_product = 1
    profits = np.empty(0)
    dates = np.empty(0)
    closes = np.empty(0)
    for i in range(len(last_trains)):
        ep.last_train = last_trains[i]
        ep.start_time = start_times[i]
        ep.end_time = end_times[i]

        ep.df_pred_path = ep.last_train + '/kospi200f_60M_pred.csv'
        ep.df_raw_path = ep.last_train + '/kospi200f_60M_raw.csv'
        ep.result_path = ep.last_train + '/pred_83_results.csv'

        p = ep.predict()
        print(last_trains[i] + " 수익률: " + str(p))
        profit_sum.append(p)#max(p, 0.5)
        profit_product *= p#max(p, 0.5)

        df = pd.read_csv(ep.result_path, encoding='euc-kr')
        profit = df['profit'].values - df['fee'].values

        profits = np.concatenate([profits, profit], axis=0)
        dates = np.concatenate([dates, df['date'].values], axis=0)
        closes = np.concatenate([closes, df['close'].values], axis=0)

    profits_rates = profits.cumsum() / (closes.mean() * 1.25 * 250000 * 0.075) + 1#profits.cumsum() / 8000000 + 1
    close_rates = (np.array(closes) - closes[0]) / closes[0] / 0.075#(np.array(closes) - closes[0]) * 250000 / 8000000 + 1

    print("평균: ", str(np.array(profit_sum).mean()))#str(profit_sum/len(last_trains)))
    print("복리누적: " + str(profit_product))


    return dates, closes, close_rates, profits, profits_rates, np.array(profit_sum).std(), profit_product, ensemble_models, path

def create_ensembles():
    print(datetime.datetime.now())

    if not os.path.isfile(result_path):
        results = []
    else:
        results = pd.read_csv(result_path).values.tolist()

    if not os.path.isfile(total_ensemble_path):
        total_ensembles = []
    else:
        total_ensembles = pd.read_csv(total_ensemble_path).values.tolist()

    best_profit_rates = 0
    best_ensemble = ''
    for i in range(14190):
        ensembles = random.sample([j for j in range(120)], ep.selected_num)
        while sorted(ensembles) in total_ensembles:
            ensembles = random.sample([j for j in range(120)], ep.selected_num)
        ensembles = sorted(ensembles)
        total_ensembles.append(ensembles)

        ep.selected_ensemble_types = list(ensembles)

        dates, closes, close_rates, profits, profits_rates, std, profit_product, ensemble_models, path = main()

        if profits_rates[-1] > best_profit_rates:
            best_profit_rates = profits_rates[-1]
            best_ensemble = ensemble_models

            dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profits_rates}
            pd.DataFrame(dic).to_excel(path, index=False, encoding='euc-kr')

        print("best profit rates: " + str(best_profit_rates) + ", " + best_ensemble)

        print(ensembles, profits_rates[-1], std, profit_product)
        results.append([ensembles[0], ensembles[1], ensembles[2], profits_rates[-1], std, profit_product])

        #if i % 10 == 0:
        print("========================================================================")
        print(" ")
        print(i)
        print(" ")
        print("========================================================================")
        pd.DataFrame(np.array(total_ensembles)).to_csv(total_ensemble_path, index=False)
        pd.DataFrame(np.array(results),
                     columns=['0', '1', '2', 'profit_rates', 'std', 'profit_product']).to_csv(result_path,
                                                                                                      index=False,
                                                                                                      encoding='euc-kr')
    print(datetime.datetime.now())
    pd.DataFrame(np.array(results), columns=['0', '1', '2', 'profit_rates', 'std', 'profit_product']).to_csv(result_path, index=False, encoding='euc-kr')

if __name__ == "__main__":
    create_ensembles()
    sys.exit(0)