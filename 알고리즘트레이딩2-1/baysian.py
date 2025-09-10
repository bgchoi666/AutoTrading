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
import baysian_losscut as bl

def ensemble_predict(reinfo, term, width):

    conf.reinfo_th = reinfo
    conf.pred_term = int(term)
    conf.reinfo_width = int(width)

    conf.result_path = conf.last_train + "/pred_83_results.csv"
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

        ['10C', '15P', '25HL']

    ]

    data.set_start_end_time(conf, '2024/05/01/09:00', '2024/05/20/15:00', '2024-04-30')

    if ep_proc:
        ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times,
                                                                           ep.end_times, "")

    results_path = "bayesian/bayesian_results_all.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).values.tolist()
    else:
        results = []

    for ensemble in ensembles:

        data.set_ensemble(conf, ensemble)
        conf.target_type = 'C'
        conf.loss_cut = 1

        best = best_params()

        model = ensemble + [round(best['params']['reinfo'], 4), int(best['params']['term']), int(best['params']['width'])]
        bl.ep_proc = ep_proc
        best_losscut = bl.create_losscut(ep_proc, model, conf.start_time, conf.end_time, conf.last_train)
        model.append(round(best_losscut, 6))

        results.append([conf.start_time, conf.end_time, conf.last_train, str(model)])

        print(model)

        pd.DataFrame(np.array(results), columns=['start', 'end', 'last_train', 'model']).to_csv(results_path, index=False)