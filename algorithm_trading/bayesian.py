# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 앙상블 예측 모델의 주어진 기간에서 최적의 reinfo_th, pred_term, loss-cut을 찾는다.

from bayes_opt import BayesianOptimization
import ensemble_test as et

import datetime

def ensemble_predict(reinfo, term, loss_cut):

    et.reinfo_th = reinfo
    et.model_reinfo_th = reinfo
    et.pred_term = int(term)
    et.loss_cut = loss_cut

    print('data processing start...')
    now = datetime.datetime.now()
    print(now)
    et.preprocessing()
    print('data processing end...')
    r = et.predict()
    print(r)

    return(r)

def best_params():
    # 파라미터의 타입을 설정합니다.
    param_space = {
        "reinfo": (0, 1),
        "term": (1, 40),
        "loss_cut": (0.005, 1),
    }

    # 베이지안 최적화를 수행합니다.
    bo = BayesianOptimization(
        f=ensemble_predict,
        pbounds=param_space,
        random_state=0,
    )
    bo.maximize(init_points=10, n_iter=10)

    return bo.max

if __name__ == "__main__":

    et.ensembles = ['25C', '5P', '30C']

    et.start_time = '2023/08/16/09:00'
    et.end_time ='2023/08/31/15:00'
    et.last_train = '2023-08-15'

    df0_path = 'kospi200f_11_60M.csv'
    df_pred_path = et.last_train + '/kospi200f_60M_pred.csv'
    df_raw_path = et.last_train + '/kospi200f_60M_raw.csv'
    result_path = et.last_train + '/pred_83_results.csv'

    best = best_params()

    print(best)