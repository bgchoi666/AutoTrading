# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# 예) 2017-12-31 ~ 2021-12-31 의 trained 모델들의 15일 간격 앙상블 수익률 list

import make_model as tm
import profit
import ensemble_test as et
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import openpyxl
import pandas as pd
import numpy as np
from datetime import datetime
import os

last_trains = [
    '2018-12-31',

    '2019-01-15', '2019-01-31', '2019-02-15', '2019-02-28',
    '2019-03-15', '2019-03-31', '2019-04-15', '2019-04-30',
    '2019-05-15', '2019-05-31', '2019-06-15', '2019-06-30',
    '2019-07-15', '2019-07-31', '2019-08-15', '2019-08-31',
    '2019-09-15', '2019-09-30', '2019-10-15', '2019-10-31',
    '2019-11-15', '2019-11-30', '2019-12-15', '2019-12-31',

    '2020-01-15', '2020-01-31', '2020-02-15', '2020-02-28',
    '2020-03-15', '2020-03-31', '2020-04-15', '2020-04-30',
    '2020-05-15', '2020-05-31', '2020-06-15', '2020-06-30',
    '2020-07-15', '2020-07-31', '2020-08-15', '2020-08-31',
    '2020-09-15', '2020-09-30', '2020-10-15', '2020-10-31',
    '2020-11-15', '2020-11-30', '2020-12-15',
#]
#last_trains = [
    '2020-12-31',
    '2021-01-15', '2021-01-31', '2021-02-15', '2021-02-28',
    '2021-03-15', '2021-03-31', '2021-04-15', '2021-04-30',
    '2021-05-15', '2021-05-31', '2021-06-15', '2021-06-30',
    '2021-07-15', '2021-07-31', '2021-08-15', '2021-08-31',
    '2021-09-15', '2021-09-30', '2021-10-15', '2021-10-31',
    '2021-11-15', '2021-11-30', '2021-12-15',
]
last_trains = [
    '2021-12-31',

    '2022-01-15', '2022-01-31', '2022-02-15', '2022-02-28',
    '2022-03-15', '2022-03-31', '2022-04-15', '2022-04-30',
    '2022-05-15', '2022-05-31', '2022-06-15', '2022-06-30',
    '2022-07-15', '2022-07-31', '2022-08-15', '2022-08-31',
    '2022-09-15', '2022-09-30', '2022-10-15', '2022-10-31',
    '2022-11-15']

start_times = [

    '2019/01/01/09:00', '2019/01/16/09:00', '2019/02/01/09:00', '2019/02/16/09:00',
    '2019/03/01/09:00', '2019/03/16/09:00', '2019/04/01/09:00', '2019/04/16/09:00',
    '2019/05/01/09:00', '2019/05/16/09:00', '2019/06/01/09:00', '2019/06/16/09:00',
    '2019/07/01/09:00', '2019/07/16/09:00', '2019/08/01/09:00', '2019/08/16/09:00',
    '2019/09/01/09:00', '2019/09/16/09:00', '2019/10/01/09:00', '2019/10/16/09:00',
    '2019/11/01/09:00', '2019/11/16/09:00', '2019/12/01/09:00', '2019/12/16/09:00',

    '2020/01/01/09:00', '2020/01/16/09:00', '2020/02/01/09:00', '2020/02/16/09:00',
    '2020/03/01/09:00', '2020/03/16/09:00', '2020/04/01/09:00', '2020/04/16/09:00',
    '2020/05/01/09:00', '2020/05/16/09:00', '2020/06/01/09:00', '2020/06/16/09:00',
    '2020/07/01/09:00', '2020/07/16/09:00', '2020/08/01/09:00', '2020/08/16/09:00',
    '2020/09/01/09:00', '2020/09/16/09:00', '2020/10/01/09:00', '2020/10/16/09:00',
    '2020/11/01/09:00', '2020/11/16/09:00', '2020/12/01/09:00', '2020/12/16/09:00',
#]
#start_times = [
    '2021/01/01/09:00', '2021/01/16/09:00', '2021/02/01/09:00', '2021/02/16/09:00',
    '2021/03/01/09:00', '2021/03/16/09:00', '2021/04/01/09:00', '2021/04/16/09:00',
    '2021/05/01/09:00', '2021/05/16/09:00', '2021/06/01/09:00', '2021/06/16/09:00',
    '2021/07/01/09:00', '2021/07/16/09:00', '2021/08/01/09:00', '2021/08/16/09:00',
    '2021/09/01/09:00', '2021/09/16/09:00', '2021/10/01/09:00', '2021/10/16/09:00',
    '2021/11/01/09:00', '2021/11/16/09:00', '2021/12/01/09:00', '2021/12/16/09:00',
]
start_times = [

    '2022/01/01/09:00', '2022/01/16/09:00', '2022/02/01/09:00', '2022/02/16/09:00',
    '2022/03/01/09:00', '2022/03/16/09:00', '2022/04/01/09:00', '2022/04/16/09:00',
    '2022/05/01/09:00', '2022/05/16/09:00', '2022/06/01/09:00', '2022/06/16/09:00',
    '2022/07/01/09:00', '2022/07/16/09:00', '2022/08/01/09:00', '2022/08/16/09:00',
    '2022/09/01/09:00', '2022/09/16/09:00', '2022/10/01/09:00', '2022/10/16/09:00',
    '2022/11/01/09:00', '2022/11/16/09:00', ]

end_times = [

    '2019/01/15/15:00', '2019/01/31/15:00', '2019/02/15/15:00', '2019/02/28/15:00',
    '2019/03/15/15:00', '2019/03/31/15:00', '2019/04/15/15:00', '2019/04/30/15:00',
    '2019/05/15/15:00', '2019/05/31/15:00', '2019/06/15/15:00', '2019/06/30/15:00',
    '2019/07/15/15:00', '2019/07/31/15:00', '2019/08/15/15:00', '2019/08/31/15:00',
    '2019/09/15/15:00', '2019/09/30/15:00', '2019/10/15/15:00', '2019/10/31/15:00',
    '2019/11/15/15:00', '2019/11/30/15:00', '2019/12/15/15:00', '2019/12/31/15:00',

    '2020/01/15/15:00', '2020/01/31/15:00', '2020/02/15/15:00', '2020/02/28/15:00',
    '2020/03/15/15:00', '2020/03/31/15:00', '2020/04/15/15:00', '2020/04/30/15:00',
    '2020/05/15/15:00', '2020/05/31/15:00', '2020/06/15/15:00', '2020/06/30/15:00',
    '2020/07/15/15:00', '2020/07/31/15:00', '2020/08/15/15:00', '2020/08/31/15:00',
    '2020/09/15/15:00', '2020/09/30/15:00', '2020/10/15/15:00', '2020/10/31/15:00',
    '2020/11/15/15:00', '2020/11/30/15:00', '2020/12/15/15:00', '2020/12/31/15:00',
#]
#end_times = [
    '2021/01/15/15:00', '2021/01/31/15:00', '2021/02/15/15:00', '2021/02/28/15:00',
    '2021/03/15/15:00', '2021/03/31/15:00', '2021/04/15/15:00', '2021/04/30/15:00',
    '2021/05/15/15:00', '2021/05/31/15:00', '2021/06/15/15:00', '2021/06/30/15:00',
    '2021/07/15/15:00', '2021/07/31/15:00', '2021/08/15/15:00', '2021/08/31/15:00',
    '2021/09/15/15:00', '2021/09/30/15:00', '2021/10/15/15:00', '2021/10/31/15:00',
    '2021/11/15/15:00', '2021/11/30/15:00', '2021/12/15/15:00', '2021/12/31/15:00',
]
end_times = [
    
    '2022/01/15/15:00', '2022/01/31/15:00', '2022/02/15/15:00', '2022/02/28/15:00',
    '2022/03/15/15:00', '2022/03/31/15:00', '2022/04/15/15:00', '2022/04/30/15:00',
    '2022/05/15/15:00', '2022/05/31/15:00', '2022/06/15/15:00', '2022/06/30/15:00',
    '2022/07/15/15:00', '2022/07/31/15:00', '2022/08/15/15:00', '2022/08/31/15:00',
    '2022/09/15/15:00', '2022/09/30/15:00', '2022/10/15/15:00', '2022/10/31/15:00',
    '2022/11/15/15:00', '2022/11/18/15:00',]


length = len(last_trains)

et.trading_9h = False
et.profit.loss_cut = 0.01
#import make_reinfo_updown as mr
#et.mr = mr
#et.tm.mr = mr

term = "2022-01-01~2022-11-18"

folder = "best_"+term+"_losscut"+str(et.profit.loss_cut)+"/"
if not os.path.isdir(folder):
    os.makedirs(folder)

def main():

    ensemble_models = ''
    for i in range(et.selected_num):
        ensemble_models += et.selected_model_types[i] + "_"

    if et.trading_9h:
        path = "9시시가에거래_ensemble_" + str(et.pred_term) + "_reinfo_" + str(et.reinfo_th) + "_" + str(et.model_reinfo_th) + "_" + ensemble_models + ".xlsx"
    else:
        path = "9시거래없음_ensemble_" + str(et.pred_term) + "_reinfo_" + str(et.reinfo_th) + "_" + str(et.model_reinfo_th) + "_" + ensemble_models + ".xlsx"

    path = folder + path

    # 모델별 각 구간별 수익률 계산, 저장
    profit_sum = 0
    profit_product = 1
    profits = np.empty(0)
    dates = np.empty(0)
    closes = np.empty(0)
    for i in range(len(last_trains)):
        et.last_train = last_trains[i]
        et.start_time = start_times[i]
        et.end_time = end_times[i]

        et.weights = [1 for i in range(et.selected_num)]
        et.selected_checkpoint_path = ['' for i in range(et.selected_num)]
        for j in range(et.selected_num):
            et.selected_checkpoint_path[j] = et.last_train + "/60M_" +et.selected_model_types[j] + "_best"

        et.df_pred_path = et.last_train + '/kospi200f_60M_pred.csv'
        et.df_raw_path = et.last_train + '/kospi200f_60M_raw.csv'
        et.result_path = et.last_train + '/pred_83_results.csv'

        #for j in range(et.selected_num):
        #    et.selected_checkpoint_path[j] = et.last_train + "/60M_" + df.values[j, i] + "_best"
        #    et.selected_checkpoint_path[j] = et.last_train + "/60M_" + model_pools[j] + "_best"

        et.preprocessing()
        p = et.predict()
        print(last_trains[i] + " 수익률: " + str(p))
        profit_sum += p#max(p, 0.5)
        profit_product *= p#max(p, 0.5)

        df = pd.read_csv(et.result_path, encoding='euc-kr')
        profit = df['profit'].values - df['fee'].values
        #if sum(profit) / 8000000 + 1 < 0.5:
        #    profit[:] = 0
        #    profit[len(profit)-1] = -4000000
        profits = np.concatenate([profits, profit], axis=0)
        dates = np.concatenate([dates, df['date'].values], axis=0)
        closes = np.concatenate([closes, df['close'].values], axis=0)

    profit_rates = profits.cumsum() / (closes.mean()*1.25*250000*0.075) + 1 #8000000 + 1
    closes = (np.array(closes) - closes[0])/(closes.mean()*1.25*0.075) + 1 #*250000/8000000 + 1

    print("평균: " + str(profit_sum/len(last_trains)))
    print("복리누적: " + str(profit_product))
    print("profit rate: " + str(profit_rates[-1]))

    dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
    pd.DataFrame(dic).to_excel(path, index=False, encoding='euc-kr')

    print(path)

    return dates, closes, profit_rates, [path, profit_sum/length, profit_product], ensemble_models

if __name__ == '__main__':

    best = [0, 0, 0, '']
    for th in [0.4, 0.5, 0.6]:
        avg_profit_rates = 0
        avg_profit_sum = 0
        avg_profit_product = 0

        pred_terms = [2]

        for pred_term in pred_terms:

            ensembles = [['3', '6', '8'], ['6', '8', '9'], ['6', '7', '8'], ['1', '3', '4'], ['4', '9', '10']]

            for enm in ensembles:

                et.selected_model_types = enm
                et.selected_num = len(et.selected_model_types)
                et.reinfo_th = th
                et.model_reinfo_th = th
                et.pred_term = pred_term

                dates, closes, profit_rates, profits, ensemble_models = main()

                path = 'best_' + term + '_losscut' + str(et.profit.loss_cut) + '.csv'
                columns = ['reinfo', 'pred_term', 'ensemble', 'profit_rates', 'profit_sum', 'profit_product']
                if os.path.isfile(path):
                    results = pd.read_csv(path).values.tolist()
                else:
                    results = []

                results.append([th, pred_term, ensemble_models, profit_rates[-1], profits[1], profits[2]])

                if profit_rates[-1] > best[0]:
                    best[0] = profit_rates[-1]
                    best[1] = th
                    best[2] = pred_term
                    best[3] = ensemble_models

                print("중간 best >>>>")
                print(best)

                print("self-reflection " + str(th) + ", pred_term " + str(pred_term) + "_" + ensemble_models + " 결과")
                print("profit rates: " + str(profit_rates[-1]))

                pd.DataFrame(np.array(results), columns=columns).to_csv(path, index=False)

    print("최종 best >>>>")
    print(best)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, closes, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = "ensemble " + ensemble_models
    plt.plot(dates, profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()