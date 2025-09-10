# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 15일 단위 수익률 생성
# 예) 2017-12-31 ~ 2021-12-31 의 trained 모델들의 15일 간격 앙상블 수익률 list

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

import make_reinfo as mr
import make_reinfo2 as mr2
import make_reinfo3 as mr3

hit_ratios = [0.2, 0.3, 0.4, 0.5]
eval_terms = [10, 20, 30, 40]
eval_widths = [10, 20, 30, 40, 50, 60, 70]

# 고정된 랜덤 앙상블 조합에 대한 주어진 기간에서의 수익률 조사를 원하는 경수 사용
random_ensemble = random.sample(conf.model_pools, 3)
random_model = [random_ensemble[0], random_ensemble[1], random_ensemble[2], random.sample(hit_ratios, 1)[0], random.sample(eval_terms, 1)[0]]

models = [

["10P", "25HL", "5C", 0.05, 10, 65, 0.005]


#random', 'model

]

conf.start_time = '2025/07/05/09:00'
conf.end_time = '2025/07/25/15:00'

conf.gubun = 0
conf.loss_cut = 0.005

ep.mr = mr3

ep.every_term_random = False
random_cnt = 10
ep.bayesian = False

ep.margin = 10000000
ep.profit.margin = 10000000

if ep.every_term_random:
    add_txt = "_every_term_random"
else:
    add_txt = "_천만투자_07월26일"

def create_all_term_ensemble3():

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    last_trains = [value for value in ep.last_trains if value < "2019-12-31"] + [value for value in ep.last_trains if value > "2020-12-15"]
    start_times = [value for value in ep.start_times if value < "2020/01/01/09:00"] + [value for value in ep.start_times if value > "2020/12/16/09:00"]
    end_times = [value for value in ep.end_times if value < "2020/01/15/15:00"] + [value for value in ep.end_times if value > "2020/12/31/15:00"]
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.folder, ep.result_path = ep.set_path("앙상블실험3", "test_eval_reflection3")

    if os.path.isfile(ep.result_path):
        results = pd.read_csv(ep.result_path).values.tolist()
    else:
        results = []

    best = [0, 0, 0, 0, 0, '']

    if not ep.every_term_random:
        n = 0
        for enm in models:
            conf.reinfo_th = enm[3]
            conf.model_reinfo_th = enm[3]
            conf.selected_model_types = enm[:3]
            conf.pred_term = enm[4]
            conf.reinfo_width = enm[5]
            if len(enm) > 6:
                conf.loss_cut = enm[6]

            ensemble_models = ""
            for i in range(conf.selected_num):
                ensemble_models = ensemble_models + conf.selected_model_types[i] + "_"
            if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term), str(conf.reinfo_width),
                                      str(conf.loss_cut)] in np.array(results)[:, :5].tolist():
                continue

            dates, closes, profits, close_rates, profit_rates, MDD, profit_product, ensemble_models, path = ep.main(conf, ep)

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                            profit_rates[-1], MDD, profit_product])
            print([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                   profit_rates[-1], MDD, profit_product])

            pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD',
                                  'profit_product']).to_csv(ep.result_path, index=False)

            if profit_rates[-1] > best[0]:
                best[0] = profit_rates[-1]
                best[1] = conf.reinfo_th
                best[2] = conf.pred_term
                best[3] = conf.reinfo_width
                best[4] = conf.loss_cut
                best[5] = ensemble_models

                #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
                #pd.DataFrame(dic).to_excel(path, index=False)

            """
            if n == 0:
                profit_df = pd.DataFrame({'dates': dates, 'model0': profit_rates})
            else:
                profit_df2 = pd.DataFrame({'model'+str(n): profit_rates})
                profit_df = pd.concat([profit_df, profit_df2], axis=1)
            n += 1
            """

            print("================= 중간 best >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(best, len(results))

        #profit_df.to_csv("ensemble_profits_2024-01-01~2024-02-07_알4-1.csv", index=False, encoding='euc-kr')

    else:

        for i in range(random_cnt):

            dates, closes, profits, close_rates, profit_rates, MDD, profit_product, ensemble_models, path = ep.main(conf, ep)

            if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term), str(conf.reinfo_width),
                                      str(round(conf.loss_cut, 4))] in np.array(results)[:, :5].tolist():
                continue

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                            profit_rates[-1], MDD, profit_product])
            print([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                   profit_rates[-1], MDD, profit_product])

            pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD',
                                  'profit_product']).to_csv(ep.result_path, index=False)

            if profit_rates[-1] > best[0]:
                best[0] = profit_rates[-1]
                best[1] = conf.reinfo_th
                best[2] = conf.pred_term
                best[3] = conf.reinfo_width
                best[4] = conf.loss_cut
                best[5] = ensemble_models

                # dic = {'dates': dates, 'profits': profit_sum, 'closes': closes, 'profit_rates': profit_rates}
                # pd.DataFrame(dic).to_excel(path, index=False, encoding='euc-kr')

            print("================= 중간 best >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(best)

    print("최종 best >>>>")
    print(best)

    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = "ensemble " + ensemble_models
    plt.plot(dates, profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()
    """

    return np.array(results)

if __name__ == '__main__':

    create_all_term_ensemble3()