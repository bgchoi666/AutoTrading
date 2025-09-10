# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 앙상블 예측 모델의 주어진 기간에서 최적의 reinfo_th, pred_term, reinfo_width, loss-cut을 찾는다.

from bayes_opt import BayesianOptimization

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import datetime
import os

import make_reinfo2 as mr2

global ep_proc

def ensemble_predict(loss_cut):

    conf.result_path = conf.last_train + "/pred_83_results.csv"
    conf.gubun = 0
    conf.loss_cut = loss_cut

    ep.mr = mr2

    if not ep_proc:

        print('data processing start...')
        now = datetime.datetime.now()
        print(now)
        data.preprocessing(conf)
        print('data processing end...')
        r = ep.predict(conf)
        print(r, conf.loss_cut, conf.selected_model_types, conf.pred_term, conf.reinfo_th, conf.reinfo_width)

        return r
    else:
        _, _, _, _, _, MDD, profit_product, _, _ = ep.main(conf, ep)

        return profit_product

def best_params():
    # 파라미터의 타입을 설정합니다.
    param_space = {
        "loss_cut": (low, high),
    }

    # 베이지안 최적화를 수행합니다.
    bo = BayesianOptimization(
        f=ensemble_predict,
        pbounds=param_space,
        random_state=0,
    )
    bo.maximize(init_points=10, n_iter=20)

    return bo.max

def create_losscut(ensembles, start, end, last_train):

    data.set_start_end_time(conf, start, end, last_train)

    if ep_proc:
        ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times,
                                                                            ep.end_times, "_reinfo2")

    results_path = "bayesian_results_losscut.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).values.tolist()
    else:
        results = []

    best_target = 0
    best_losscut = 1
    best_ensemble = []
    for ensemble in ensembles:

        data.set_target_type(conf, 'C')
        data.set_pred_term(conf, ensemble[conf.selected_num+1])
        data.set_reinfo(conf, ensemble[conf.selected_num])  # reinfo_th,
        data.set_ensemble(conf, ensemble[:conf.selected_num])
        data.set_width(conf, ensemble[conf.selected_num+2])  # eval_width

        best = best_params()

        if best_target < best['target']:
            best_target = best['target']
            best_losscut = best['params']['loss_cut']
            best_ensemble = ensemble

    best_ensemble.append(round(best_losscut, 6))
    results.append([start, end, conf.last_train, str(best_ensemble)])
    pd.DataFrame(results, columns=['start', 'end', 'last_train', 'model']).to_csv(results_path, index=False)
    return best_ensemble, best_target, best_losscut

if __name__ == "__main__":

    global ep_proc
    ep_proc = False

    ensembles = [

        ['10P', '20P', '5P', 0.1, 40, 5, 0.005]

    ]

    start_time = '2025/01/01/09:00'
    end_time = '2025/01/31/15:00'
    last_train = '2025-01-31'

    low = 0.001
    high = 0.01

    losscut = create_losscut(ensembles, start_time, end_time, last_train)
    print("best_losscut: ", str(losscut) )