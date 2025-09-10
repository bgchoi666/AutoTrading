# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# reinfo, pred_term은 [0, 1], [1, 40]에서 random으로 n번 반복
# 예) 2017-12-31 ~ 2021-12-31 의 trained 모델들의 앙상블 수익률 list

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

import make_reinfo as mr
import make_reinfo2 as mr2

import re

best_ensemble_path = 'best_ensemble_2020-12-31~2022-11-30.csv'
best_test_path = 'best_test_results_2021-01-31~2022-12-31.csv'

if __name__ == "__main__":
    df = pd.read_csv(best_ensemble_path)

    results = []
    for i in range(len(df)):
        test_index = ep.end_times.index(df['term'].values[i]) + 1
        conf.start_time = ep.start_times[test_index]
        conf.end_time = ep.end_times[test_index]
        conf.last_train = ep.last_trains[test_index]

        data.set_path(conf)
        data.set_target_type(conf, 'C')

        conf.gubun = 0
        ensemble = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", df['ensemble'].values[i]).split(' ')[:3]
        data.set_pred_term(conf, df['pred_term'].values[i])
        data.set_reinfo(conf, df['self-reflection'].values[i])  # reinfo
        data.set_ensemble(conf, ensemble)  # selected_model_types
        data.set_profit(conf, 0.01, 1)  # loss_cut, profit_cut
        conf.reinfo_width = df['width'].values[i]

        print("preprocessing start......")
        data.preprocessing(conf)
        print("preprocessing end........")

        # type=0 --> 0: 중립 1:고점 2:저점,  type=1 --> n일 후 0:하락  1: 상승
        r = ep.predict(conf)

        results.append([conf.start_time+"~"+conf.end_time, conf.selected_model_types, conf.reinfo_th, conf.pred_term, conf.reinfo_width, str(r)])
        pd.DataFrame(np.array(results), columns=['term', 'ensemble', 'self-reflection', 'pred_term', 'width', 'profit']).to_csv(best_test_path, index=False)


