# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 거래에 실재 적용되는 앙상블 모델의 prediction 결과와 수익률

import data
from data import config
conf = config()
import ensemble_proc as ep
import profit

import pandas as pd
import datetime

import make_reinfo2 as mr
ep.mr = mr

# 04월 09일
data.set_start_end_time(conf, '2024/04/01/09:00', '2024/04/09/15:00', '2024-03-31')
data.set_path(conf)
data.set_target_type(conf, 'C')
data.set_pred_term(conf, 17)
data.set_reinfo(conf,  0.5) # reinfo
data.set_ensemble(conf, ['25P', '30P', '5HL']) # selected_model_types
data.set_profit(conf, 0.005, 1) # loss_cut, profit_cut
conf.reinfo_width = 63


if __name__ == "__main__":

    conf.end_time = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    conf.start_time = (datetime.datetime.strptime(conf.last_train, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime("%Y/%m/%d/09:00")

    #conf.end_time = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    #conf.start_time = (datetime.datetime.strptime(conf.end_time, '%Y/%m/%d/%H:%M') - datetime.timedelta(days=15)).strftime("%Y/%m/%d/09:00")

    #prediction only
    conf.gubun = 0
    conf.result_path = 'pred_88_results2.csv'

    print('data processing start...')
    now = datetime.datetime.now()
    print(now)
    data.preprocessing(conf)
    print('data processing end...')
    r = ep.predict(conf)
    print(r)

    print(pd.read_csv(conf.result_path).values[-7:, [0, 1, 5]])

    profit.loss_cut = conf.loss_cut
    profit.result_path = conf.result_path
    rate = profit.calc_profit()
    print(rate)
