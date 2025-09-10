# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# 예) 2017-12-31 ~ 2021-12-31 의 trained 모델들의 앙상블 거래마다 누적 수익률  기록 list

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


models = [

["15HL", "20P", "25C"],
["10P", "25C", "5C"],
["15HL", "20C", "30C"],
["15P", "25P", "5HL"],
["15HL", "30C", "5HL"],
["20P", "25C", "5P"],
["10P", "30C", "5C"],
["10C", "10HL", "5HL"],
["20HL", "40C", "5C"],
["25HL", "30C", "5HL"],

]


if __name__ == '__main__':

    hit_ratios = [0.29]
    eval_terms = [32]
    conf.reinfo_width = 47

    conf.start_time = '2017/01/01/09:00'
    conf.end_time = '2022/12/31/15:00'

    conf.gubun = 0

    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times, ep.end_times, "")
    ep.folder, ep.result_path = ep.set_path("앙상블실험", "eval_reflection")

    if os.path.isfile(ep.result_path):
        results = pd.read_csv(ep.result_path).values.tolist()
    else:
        results = []

    best = [0, 0, 0, '']
    for th in hit_ratios:
        conf.reinfo_th = th

        for pred_term in eval_terms:
            conf.pred_term = pred_term

            for enm in models:
                conf.selected_model_types = enm

                ensemble_models = ""
                for i in range(conf.selected_num):
                    ensemble_models = ensemble_models + conf.selected_model_types[i] + "_"
                if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term)] in np.array(results)[:, :3].tolist():
                    continue

                dates, closes, profits, close_rates, profit_rates, std, profit_product, ensemble_models, path = ep.main(conf, ep)

                results.append([ensemble_models, th, pred_term, profit_rates[-1], std, profit_product])
                print([ensemble_models, th, pred_term, profit_rates[-1], std, profit_product])

                pd.DataFrame(np.array(results),
                             columns=['ensemble', 'self-reflection', 'pred_term', 'profit_rates', 'std',
                                      'profit_product']).to_csv(ep.result_path, index=False)

                if profit_rates[-1] > best[0]:
                    best[0] = profit_rates[-1]
                    best[1] = th
                    best[2] = pred_term
                    best[3] = enm

                print("================= 중간 best >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(best)

    print("최종 best >>>>")
    print(best)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = "ensemble " + ensemble_models
    plt.plot(dates, profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()