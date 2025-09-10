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

def ensemble_predict(reinfo, term, width):

    conf.reinfo_th = reinfo
    conf.pred_term = term
    conf.reinfo_width = int(width)

    conf.result_path = conf.last_train + "/pred_88_results.csv"
    conf.gubun = 0

    ep.mr = mr2

    if not ep_proc:
        print('data processing start...')
        now = datetime.datetime.now()
        print(now)
        data.preprocessing(conf)
        print('data processing end...')
        r = ep.predict(conf)
        print(r, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.selected_model_types, conf.selected_checkpoint_path)

        return r
    else:
        _, _, _, _, _, MDD, profit_product, _, _ = ep.main(conf, ep)

        return profit_product / MDD

def best_params():
    # 파라미터의 타입을 설정합니다.
    param_space = {
        "reinfo": (0, 0.5),
        "term": (5, 41),
        "width": (10, 77),
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

        ['25HL', '40HL', '5P'],

    ]

    data.set_start_end_time(conf, '2024/01/16/09:00', '2024/01/31/15:00', '2024-01-15')
    start_time = conf.start_time

    if ep_proc:
        ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times,
                                                                           ep.end_times, "")

    results_path = "bayesian/bayesian_results.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).values.tolist()
    else:
        results = []

    for ensemble in ensembles:

        data.set_ensemble(conf, ensemble)
        conf.target_type = 'C'
        conf.loss_cut = 0.005

        best = best_params()
        results.append([start_time + "~" + conf.end_time, str(ensemble), best['target'], best['params']['reinfo'], best['params']['term'], best['params']['width']])

        print(ensemble, best)

        pd.DataFrame(np.array(results), columns=['date', 'ensemble', 'profit_product', 'reinfo', 'pred_term', 'width']).\
            to_csv(results_path, index=False)