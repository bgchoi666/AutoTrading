# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# reinfo, pred_term은 [0, 1], [1, 40]에서 random으로 n번 반복
# 예) 2017-12-31 ~ 2021-12-31 의 trained 모델들의 앙상블 수익률 list

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

ensembles = [

    ['20HL', '25P', '30C']

]
random_ensemble = True

cnt = 100


if __name__ == "__main__":
    start_times = ep.start_times.copy()
    end_times = ep.end_times.copy()
    last_trains = ep.last_trains.copy()

    start_time = '2021/09/16/09:00'
    end_time = '2022/09/30/15:00'

    start_index = start_times.index(start_time)
    end_index = end_times.index(end_time)

    best_ensemble_path = 'best_ensemble_2021-01-01~2022-12-15.csv'

    if not os.path.isfile(best_ensemble_path):
        best_ensemble = []
    else:
        best_ensemble = pd.read_csv(best_ensemble_path).values.tolist()

    conf.start_time = start_time
    conf.end_time = end_times[start_index+23]
    while conf.end_time <= end_time:

        conf.gubun = 0

        ep.mr = mr2
        add_txt = ""

        ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
        ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "eval_reflection_random")

        results = []
        best = [0, 0, 0, '', 0, 0, 0]
        while len(results) < cnt:

            conf.reinfo_th = random.randint(0, 32) / 100# #random.sample([0.2, 0.3, 0.4, 0.5], 1)[0]
            conf.pred_term = random.randrange(5, 41, 5) #random.sample([10, 20, 30, 40], 1)[0]
            conf.reinfo_width = random.randrange(10, 71, 10) #random.sample([10, 20, 30, 40, 50, 60, 70], 1)[0]
            conf.target_type = 'C'

            for enm in ensembles:

                if not random_ensemble:
                    conf.selected_model_types = enm
                else:
                    conf.selected_model_types = sorted(random.sample(conf.model_pools, conf.selected_num))

                ensemble_models = ""
                for i in range(conf.selected_num):
                    ensemble_models = ensemble_models +conf.selected_model_types[i] + "_"
                if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term)] in np.array(results)[:, :4].tolist():
                    continue

                dates, closes, profits, close_rates, profit_rates, MDD, profit_product, ensemble_models, parh = ep.main(conf, ep)


                if profit_rates[-1] > best[0]:
                    best[0] = profit_rates[-1]
                    best[1] = MDD
                    best[2] = profit_product
                    best[3] = conf.selected_model_types
                    best[4] = conf.reinfo_th
                    best[5] = conf.pred_term
                    best[6] = conf.reinfo_width

                print(conf.start_time, conf.end_time)
                print("중간 best >>>>")
                print(best)

                print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width: ",
                      profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term, conf.reinfo_width)

                results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, profit_rates[-1], MDD, profit_product])

                print("여기까지 . . .", len(results))

        print("최종 best >>>> ")
        print(best)

        best_ensemble.append([conf.end_time, best[3], best[4], best[5], best[6], best[0], best[1], best[2]])
        pd.DataFrame(np.array(best_ensemble),
                     columns=['term', 'ensemble', 'self-reflection', 'pred_term', 'width', 'profit_rates', 'MDD',
                              'profit_product']).to_csv(best_ensemble_path, index=False)

        start_index += 1
        conf.start_time = start_times[start_index]
        conf.end_time = end_times[start_index + 23]

    pd.DataFrame(np.array(best_ensemble), columns=['term', 'ensemble', 'self-reflection', 'pred_term', 'width', 'profit_rates', 'MDD',
                                             'profit_product']).to_csv(best_ensemble_path, index=False)