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

mode = ''

if __name__ == '__main__':

    model = ['10C', '15HL', '40P', 0.25, 6, 10, 0.005]

    data.set_target_type(conf, 'C')
    data.set_pred_term(conf, model[4])
    data.set_reinfo(conf, model[3]) # reinfo probability
    data.set_ensemble(conf, model[:3])  # selected_model_types
    data.set_width(conf, model[5]) # evaluation width
    data.set_profit(conf, model[6], 1)  # loss_cut, profit_cut

    conf.gubun = 0

    conf.start_time = '2025/01/01/09:00'
    conf.end_time = '2025/03/31/15:00'

    ep.profit.slippage = 0.05
    ep.margin = 10000000
    ep.profit.margin = 10000000

    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times, ep.end_times,
                                                  "_천만투자_" + str(conf.pred_term) + conf.target_type + "_" + str(conf.reinfo_th))
    ep.folder, ep.result_path = ep.set_path("앙상블실험", "eval_reflection")

    ep.every_term_random = False

    if os.path.isfile(ep.result_path):
        results = pd.read_csv(ep.result_path).values.tolist()
    else:
        results = []

    if mode == 'main_random':
        dates, closes, profits, close_rates, profit_rates, std, profit_product, ensemble_models, path = ep.main_random(conf, ep)
    else:
        dates, closes, profits, close_rates, profit_rates, std, profit_product, ensemble_models, path = ep.main(conf, ep)

    #results.append([conf.selected_model_types, conf.reinfo_th, conf.pred_term, profit_rates[-1], std, profit_product])

    #pd.DataFrame(np.array(results),
    #             columns=['ensemble', 'self-reflection', 'pred_term', 'profit_rates', 'std', 'profit_product']).to_csv(ep.result_path, index=False)

    #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
    #pd.DataFrame(dic).to_excel(path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = "ensemble " + ensemble_models
    plt.plot(dates, profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()