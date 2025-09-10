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

def ensemble_predict(loss_cut):

    conf.result_path = conf.last_train + "/pred_88_results.csv"
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
        "loss_cut": (0.001, 0.01),
    }

    # 베이지안 최적화를 수행합니다.
    bo = BayesianOptimization(
        f=ensemble_predict,
        pbounds=param_space,
        random_state=0,
    )
    bo.maximize(init_points=10, n_iter=20)

    return bo.max

if __name__ == "__main__":

    ep_proc = False

    ensembles = [

        ['25HL', '40HL', '5P', 0.3891, 36, 75],

    ]

    data.set_start_end_time(conf, '2024/01/16/09:00', '2024/01/31/15:00', '2024-01-15')
    start_time = conf.start_time

    if ep_proc:
        ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times,
                                                                           ep.end_times, "_reinfo2")

    results_path = "bayesian/bayesian_results_losscut.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).values.tolist()
    else:
        results = []

    for ensemble in ensembles:

        data.set_target_type(conf, 'C')
        data.set_pred_term(conf, ensemble[4])
        data.set_reinfo(conf, ensemble[3])  # reinfo_th,
        data.set_ensemble(conf, ensemble[:3])  # selected_model_types
        data.set_width(conf, ensemble[5])  # eval_width

        ensemble = conf.selected_model_types

        ensemble = ensemble[0] + "_" + ensemble[1] + "_" + ensemble[2] + "_" + str(conf.pred_term) + "_" + \
                   str(conf.reinfo_th) + "_" + str(conf.reinfo_width)

        best = best_params()
        results.append([start_time + "~" + conf.end_time, str(ensemble), best['target'], best['params']['loss_cut']])

        print(ensemble, best)

        pd.DataFrame(np.array(results), columns=['date', 'ensemble', 'rate', 'loss_cut']).to_csv(results_path, index=False)