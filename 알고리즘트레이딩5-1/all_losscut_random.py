# Copyright 2024 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 기간, ensemble 모델에 대한 최적의 losscut 탐색
# losscut은 [0, 0.01]에서 random으로 n번 반복
# 각 선택 losscut에서 대한 주어진 기간에서 수익률 기록

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import sys
from datetime import datetime

import make_reinfo as mr
import make_reinfo2 as mr2

model = ['25P', '30HL', '40P', 0.02, 30, 10]

data.set_start_end_time(conf, '2024/07/01/09:00', '2024/07/05/15:00', '2024-06-30')
start_time = conf.start_time

data.set_path(conf)

data.set_target_type(conf, 'C')
data.set_pred_term(conf, model[4])
data.set_reinfo(conf, model[3])  # reinfo_th,
data.set_ensemble(conf, model[:3])  # selected_model_types
data.set_width(conf, model[5])  # eval_width

# no training data
conf.gubun = 0

ep.mr = mr2
ep.profit.slippage = 0.05

ep.term = datetime.strptime(conf.start_time, "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d") + "~" + \
       datetime.strptime(conf.end_time, "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d")
ep.result_path = "eval_reflection_random/random_losscut_" + "_slippage" + str(ep.profit.slippage) + "_" + ep.term + ".csv"

print("preprocessing start......")
data.preprocessing(conf)
print("preprocessing end........")

ep.profit.slippage = 0.05

cnt = 100

if __name__ == "__main__":



    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, '', 0, 0, 0, 0]
    while len(results) < cnt:

        conf.loss_cut = random.randint(0, 100) / 10000

        ensemble_models = '["'
        for i in range(conf.selected_num - 1):
            ensemble_models = ensemble_models + conf.selected_model_types[i] + '", "'
        ensemble_models = ensemble_models + conf.selected_model_types[2] + '", ' + str(conf.reinfo_th) + ', ' + str(
            conf.pred_term) + ', ' + str(conf.reinfo_width) + ', ' + str(conf.loss_cut) + ']'

        if results != [] and ensemble_models in np.array(results)[:, 0].tolist():
            continue

        r = ep.predict(conf)


        if r > best[0]:
            best[0] = r
            best[1] = conf.selected_model_types
            best[2] = conf.reinfo_th
            best[3] = conf.pred_term
            best[4] = conf.reinfo_width
            best[5] = conf.loss_cut

            best_ensemble = ensemble_models

        print("중간 best >>>>")
        print(best)

        print("profit rates, ensemble, reinfo, pred_term, width, loss_cut: ",
              r, conf.selected_model_types, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut)

        results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut, r])
        results_df = pd.DataFrame(np.array(results), columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'loss_cut', 'profit_rates'])

        results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    results_df = pd.DataFrame(np.array(results),
                              columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'loss_cut', 'profit_rates'])
    print("평균 >>>>")
    print(results_df['profit_rates'].values.astype(dtype=np.float).mean())

    exit(0)
