# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# self-reflection reset

import pandas as pd
import numpy as np

th = 0.1
pred_term = 2

def make_target(df):
    # 고저점 예측 - 0: 고점 1: 저점

    target = []
    for i in range(len(df)):
        if i < len(df) - 1 - pred_term:
            if df.loc[i+pred_term, '종가'] >= df.loc[i, '종가']:
                target.append(1)
            else:
                target.append(0)
        else:
            if df.loc[len(df)-1, '종가'] >= df.loc[i, '종가']:
                target.append(1)
            else:
                target.append(0)

    df['target'] = target

    return target


def reinfo(pred, pred_results):

    df_results = pd.DataFrame(pred_results, columns=['date', 'results', '시가', '고가', '저가', '종가'])
    target = make_target(df_results)

    new_pred = []
    for i in range(len(pred)):
        n = i - pred_term
        cnt = 0
        for j in range(i+1):
            if j <= n and pred[j] == target[j]:
                cnt += 1
            elif j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) > 0:
                if pred[i] == 1:
                    cnt += 1
            elif j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) <= 0:
                if pred[i] == 0:
                    cnt += 1
        prob = cnt / (i + 0.00001)

        if prob < th:
            if pred[i] == 0:
                new_pred.append(1)
            else:
                new_pred.append(0)
        else:
            new_pred.append(pred[i])

    return np.array(new_pred)
