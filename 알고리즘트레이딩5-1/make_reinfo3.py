# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# the functions for self-reflected prediction

import numpy as np
import random
import pandas as pd

pred_term = 0.005
target_type = 'C'
loss_cut = 0.005

th = 0.5
ths = []

def make_train_data(df):

    # 고저점 예측 - 0: 정상  1: 고점 2: 저점 until pred_term
    target = []
    for i in range(len(df)):
        # 현재 시점과 같은 날 마지막 시점
        date = df['date'].values[i]
        last_time = date[:11] + '17:00'

        # 평가를 위한 같은날의 dataframe
        eval_df = df.loc[df['date'] >= date].loc[df['date'] <= last_time]

        if len(eval_df) == 1:
            target.append(0)
            continue

        a = float(df.loc[i, '종가'])
        b = float(eval_df['종가'].values[-1])
        c = eval_df['고가'].values[1:].astype(dtype=float).max()
        d = eval_df['저가'].values[1:].astype(dtype=float).min()

        if a > b and a*(1+loss_cut) > c:
            target.append(1)
        elif a < b and a*(1-loss_cut) < d:
            target.append(2)
        else:
            target.append(0)

    return target

def reinfo(pred, pred_results, start_time, reinfo_width):
    global th

    df = pd.DataFrame(pred_results, columns=['date', 'results', '시가', '고가', '저가', '종가'])
    target = make_train_data(df)
    start_inex = df.loc[df['date'] >= start_time].index.min()

    new_pred = []
    for i in range(start_inex, len(pred)):
        cnt = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for j in range(max(0, i-reinfo_width), i):
            if df.loc[j, 'date'][:11] != df.loc[i, 'date'][:11]:
                cnt[pred[j]][target[j]] += 1
            elif float(pred_results[i, 5]) - float(pred_results[j, 5]) > 0 and \
                    pred_results[j+1:i+1, 4].astype(dtype=float).min() > float(pred_results[j, 5]) * (1 - loss_cut):
                cnt[pred[j]][2] += 1
            elif float(pred_results[i, 5]) - float(pred_results[j, 5]) < 0 and \
                    pred_results[j+1:i+1, 3].astype(dtype=float).max() < float(pred_results[j, 5]) * (1 + loss_cut):
                cnt[pred[j]][1] += 1
            else:
                cnt[pred[j]][0] += 1

        prob = np.zeros(shape=3)
        prob[0] = cnt[0, 0] / (cnt[0, :].sum() + 0.00001)
        prob[1] = cnt[1, 1] / (cnt[1, :].sum() + 0.00001)
        prob[2] = cnt[2, 2] / (cnt[2, :].sum() + 0.00001)

        #if th == 'r':
        #    th = random.random()
        if prob[pred[i]] < th:
            new_pred.append(np.argmax(cnt[pred[i], :]))
        else:
            new_pred.append(pred[i])
    return np.array(new_pred)


