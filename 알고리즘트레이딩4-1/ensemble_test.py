# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 여러 구간에서 train된 모델들의 앙상블의 performance를 test

import data
from data import config as conf
import ensemble_proc as ep
import pandas as pd

import make_reinfo2 as mr2
import make_reinfo3 as mr3
ep.mr = mr3

model = ['15P', '40HL', '5C', 0.35, 10, 30, 0.0015]


data.set_start_end_time(conf, '2025/08/01/09:00', '2025/08/15/15:00', '2025-07-31')
start_time = conf.start_time

data.set_path(conf)
conf.result_path = "test_results.csv"

data.set_target_type(conf, 'C')
data.set_pred_term(conf, model[4])
data.set_reinfo(conf, model[3])  # reinfo_th,
data.set_ensemble(conf, model[:3])  # selected_model_types
data.set_width(conf, model[5])  # eval_width
data.set_profit(conf, model[6], 1) # loss_cut, profit_cut

ep.profit.slippage = 0.05
ep.profit.trading_9h = True

if __name__ == "__main__":

    #prediction only
    conf.gubun = 0

    print("preprocessing start......")
    data.preprocessing(conf)
    print("preprocessing end........")

    # type=0 --> 0: 중립 1:고점 2:저점,  type=1 --> n일 후 0:하락  1: 상승
    r = ep.predict(conf)

    df = pd.read_csv(conf.result_path, encoding='euc-kr')
    df = df.loc[df['date'] >= start_time]

    r = (df['profit'].values.sum() - df['fee'].values.sum()) / (df['close'].values.mean() * 250000 * 0.085) + 1

    print("수익률: " + str(r))

