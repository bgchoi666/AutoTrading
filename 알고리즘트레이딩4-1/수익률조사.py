# Copyright 2024 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# best_model.csv에 기록된 모델들에 의한 수익률

import pandas as pd
import numpy as np
import datetime

import data
from data import config as conf
import ensemble_proc as ep

import make_reinfo2 as mr2
import make_reinfo3 as mr3
ep.mr = mr3
mode = 'mr3'

last_trains = np.array(['2023-12-31',
    '2024-01-15', '2024-01-31', '2024-02-15', '2024-02-29',
    '2024-03-15', '2024-03-31', '2024-04-15', '2024-04-30',
    '2024-05-15', '2024-05-31', '2024-06-15', '2024-06-30',
    '2024-07-15', '2024-07-31', '2024-08-15', '2024-08-31',
    '2024-09-15', '2024-09-30', '2024-10-15', '2024-10-31',
    '2024-11-15', '2024-11-30', '2024-12-15',
    '2024-12-31',
    '2025-01-15', '2025-01-31', '2025-02-15', '2025-02-28',
    '2025-03-15', '2025-03-31', '2025-04-15', '2025-04-30',
    '2025-05-15', '2025-05-31', '2025-06-15', '2025-06-30',
    '2025-07-15', '2025-07-31', '2025-08-15', '2025-08-31',
    '2025-09-15', '2025-09-30', '2025-10-15', '2025-10-31',
    '2025-11-15', '2025-11-30', '2025-12-15',
    ])

first_start = '2024/12/31/09:00'
final_end = '2025/08/27/15:00'


df = pd.read_csv("H:/알고리즘트레이딩/best_model.csv", encoding='euc-kr')

if mode == 'mr2':
    df = df.loc[df['알고리즘'] == '알4-1_random3개월_6개월_12개월_mr2'].reset_index(drop=True)
    #df = df.loc[df['알고리즘'] == '알4-1_1년_2년_mr2'].reset_index(drop=True)

    #df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
else:
    df = df.loc[df['알고리즘'] == '알4-1_random3개월_6개월_12개월_mr3'].reset_index(drop=True)

for i in range(len(df)):
    df.loc[i, '최근탐색기간'] = df.loc[i, '최근탐색기간'].split('~')[1]
df = df.sort_values(by=['최근탐색기간'], axis=0).reset_index(drop=True)

df = df.loc[df['최근탐색기간'] >= datetime.datetime.strptime(first_start, "%Y/%m/%d/09:00").strftime("%Y-%m-%d")].reset_index(drop=True)
df = df.loc[df['최근탐색기간'] <= datetime.datetime.strptime(final_end, "%Y/%m/%d/15:00").strftime("%Y-%m-%d")]\
    .sort_values(by=['최근탐색기간'], axis=0).reset_index(drop=True)

total = 0
for i in range(len(df)):

    term1 = df.loc[i, '최근탐색기간']
    st = (datetime.datetime.strptime(term1, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y/%m/%d/09:00")

    if i == len(df)-1:
        en = final_end
    else:
        term2 = df.loc[i + 1, '최근탐색기간']
        en = datetime.datetime.strptime(term2, "%Y-%m-%d").strftime("%Y/%m/%d/15:00")

    if st >= en:
        continue

    indexs = np.where(last_trains <= term1)
    last_train = last_trains[indexs[-1][-1]]

    # 주어진 모델과 기간에서 수익 계산
    model = df.loc[i, 'ensemble2'].replace('"', '').replace('[', '').replace(']', '').replace('\\', '').replace(' ', '').replace('\'', '').split(',')
    if len(model) < 7:
        model.append(0.005)
    elif model[6] == 'N':
        model[6] = 0.005
    #else:
    #    model[6] = 0.005

    data.set_start_end_time(conf, st, en, last_train)
    start_time = conf.start_time

    data.set_path(conf)
    conf.result_path = 'test_results.csv'

    conftarget_type = 'C'
    conf.pred_term = int(model[4])
    conf.reinfo_th = float(model[3])  # reinfo_th,
    data.set_ensemble(conf, model[:3])  # selected_model_types
    conf.loss_cut = float(model[6])  # loss_cut, profit_cut
    conf.reinfo_width = int(model[5])  # eval_width

    ep.profit.slippage = 0.0
    #ep.profit.trading_9h = True

    # prediction only
    conf.gubun = 0

    print("preprocessing start......")
    data.preprocessing(conf)
    print("preprocessing end........")

    # type=0 --> 0: 중립 1:고점 2:저점,  type=1 --> n일 후 0:하락  1: 상승
    r = ep.predict(conf)

    r_df = pd.read_csv(conf.result_path, encoding='euc-kr')
    r_df = r_df.loc[r_df['date'] >= start_time]

    total += r_df['profit'].values.sum() - r_df['fee'].values.sum()

    print(st + "~" + en + " 수익률: " + str(r))

print(first_start + "~" + final_end + " total: " + str(total))

exit(0)