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


def ensemble_predict(reinfo):

    conf.result_path = conf.last_train + "/pred_88_results.csv"
    conf.gubun = 0
    conf.reinfo_th = reinfo

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
        "reinfo": (low, high),
    }

    # 베이지안 최적화를 수행합니다.
    bo = BayesianOptimization(
        f=ensemble_predict,
        pbounds=param_space,
        random_state=0,
    )
    bo.maximize(init_points=10, n_iter=20)

    return bo.max

def create_reinfo(model, start, end, last_train):

    ensembles = model

    data.set_start_end_time(conf, start, end, last_train)

    if ep_proc:
        ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times,
                                                                           ep.end_times, "_reinfo2")

    results_path = "bayesian_results_reinfo.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).values.tolist()
    else:
        results = []

    best_target = 0
    best_reinfo = 0
    best_ensemble = []
    for ensemble in ensembles:

        data.set_target_type(conf, 'C')
        data.set_pred_term(conf, ensemble[4])
        #data.set_reinfo(conf, ensemble[3])  # reinfo_th,
        data.set_ensemble(conf, ensemble[:3])
        data.set_width(conf, ensemble[5])  # eval_width
        conf.loss_cut = ensemble[6] # losscut

        best = best_params()

        if best_target < best['target']:
            best_target = best['target']
            best_reinfo = best['params']['reinfo']
            best_ensemble = ensemble

    best_ensemble[3] = best_reinfo
    results.append([start, end, last_train, str(best_ensemble)])

    pd.DataFrame(np.array(results), columns=['start', 'end', 'last_train', 'model']).to_csv(results_path, index=False)

    return best_ensemble, best_target, best_reinfo

if __name__ == "__main__":

    global ep_proc
    ep_proc = False

    ensembles = [

        ['15HL', '25C', '5HL', 0.15, 40, 5, 0.006235]

    ]
    start_time = '2024/11/16/09:00'
    end_time = '2024/12/20/15:00'
    last_train = '2024-12-15'

    low = 0.0
    high = 0.9

    reinfo = create_reinfo(ensembles, start_time, end_time, last_train)
    print("best_reinfo: ", str(reinfo) )