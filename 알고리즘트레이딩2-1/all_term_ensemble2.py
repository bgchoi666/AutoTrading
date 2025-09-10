# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 랜덤 선택 앙상블의 수익률 생성

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import random

import make_reinfo as mr
import make_reinfo2 as mr2

mode = 'main'

if __name__ == '__main__':

    model = ['30C', '5HL', '5P', 1, 30, 50, 0.003673]
    m = len(model)

    data.set_target_type(conf, 'C')
    data.set_pred_term(conf, model[m-3])
    data.set_reinfo(conf, model[m-4])  # reinfo_th,
    data.set_ensemble(conf, model[:m-4])  # selected_model_types
    data.set_profit(conf, model[m-1], 1)  # loss_cut, profit_cut
    data.set_width(conf, model[m-2])

    conf.gubun = 0

    ep.mr = mr2
    ep.profit.slippage = 0.05
    ep.margin = 10000000
    ep.profit.margin = 10000000

    conf.start_time = '2025/01/01/09:00'
    conf.end_time = '2025/02/28/15:00'

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times, ep.end_times,
                                                  "_천만투자_" + str(conf.pred_term) + conf.target_type + "_" + str(conf.reinfo_th))
    ep.folder, ep.result_path = ep.set_path("앙상블실험", "eval_reflection_random")

    ep.every_term_random = True
    ep.bayesian = False

    if os.path.isfile(ep.result_path):
        results = pd.read_csv(ep.result_path).values.tolist()
    else:
        results = []

    if mode == 'main_random':
        dates, closes, profits, close_rates, profit_rates, std, profit_product, ensemble_models, path = ep.main_random(conf,ep)
    else:
        dates, closes, profits, close_rates, profit_rates, std, profit_product, ensemble_models, path = ep.main(conf, ep)

    #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
    #pd.DataFrame(dic).to_excel(path, index=False)

    #results.append([et.selected_model_types, conf.reinfo_th, conf.pred_term, profit_rates[-1], std, profit_product)

    #pd.DataFrame(np.array(results),
    #             columns=['ensemble', 'self-reflection', 'pred_term', 'profit_rates', 'std', 'profit_product']).to_csv(ep.result_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = "ensemble " + ensemble_models
    plt.plot(dates, profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()