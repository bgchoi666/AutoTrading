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

import make_reinfo3 as mr3
ep.mr = mr3

# 6월 27일 the best of random mr3 1000 (random3_6_12개월), 0.9324897125
#model = ['15HL', '20C', '20P', 0.15, 10, 30, 0.0015]
# 7월 4일 the best of random mr3 1000 (random3_6_12개월), 0.8967954125000006
#model = ['10P', '15HL', '25P', 0.2, 10, 55, 0.0025]
# 6월 27일 the best of random mr3 1000 (random3개월), 1.0254281249999992
#model = ["20HL", "30HL", "30P", 0.05, 10, 70, 0.002]
# 7월 18일 the best of random mr3 1000 (random3_6_12개월), 1.0867618375
#model = ['10C', '15HL', '20HL', 0.5, 10, 45, 0.002]
# 7월 25일 the best of random mr3 1000 (random3개월)
#model = ["10HL", "15HL", "30P", 0.05, 10, 45, 0.005]#random3_6_12개월['20P', '30HL', '5HL', 0.1, 10, 15, 0.002]
# 8월 1일 the best of random mr3 1000 (random3개월)
#model = ['15P', '25C', '40P', 0.75, 10, 5, 0.0015]
# 8월 15일 the best of random mr3 1000 (random3_6_12개월)
#model = ['15C', '15HL', '5HL', 0.9, 10, 35, 0.002]

# 8월 31일 the best of random mr3 1000 (random3_6_12개월)
model = ['15HL', '20HL', '30C', 0.3, 10, 10, 0.002]

data.set_start_end_time(conf, '2025/09/01/09:00', '2025/03/26/15:00', '2025-08-31')
data.set_path(conf)
data.set_target_type(conf, 'C')
data.set_pred_term(conf, model[4])
data.set_reinfo(conf, model[3])  # reinfo_th,
data.set_ensemble(conf, model[:3])  # selected_model_types
data.set_width(conf, model[5])  # eval_width
data.set_profit(conf, model[6], 1) # loss_cut, profit_cut

ep.profit.slippage = 0.05

if __name__ == "__main__":

    #ep.profit.trading_9h = True

    conf.end_time = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    #conf.start_time = (datetime.datetime.strptime(conf.last_train, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime("%Y/%m/%d/09:00")
    #conf.start_time = (datetime.datetime.strptime(conf.end_time, '%Y/%m/%d/%H:%M') - datetime.timedelta(days=2)).strftime("%Y/%m/%d/09:00")

    #prediction only
    conf.gubun = 0

    conf.result_path = "pred_89_results.csv"

    print('data processing start...')
    now = datetime.datetime.now()
    print(now)
    data.preprocessing(conf)
    print('data processing end...')
    r = ep.predict(conf)
    print(r)

    print(pd.read_csv(conf.result_path).values[-7:, [0, 1, 5]])

