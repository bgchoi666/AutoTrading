# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# self-reflection reset

import pandas as pd
import numpy as np

pred_term = 40
target_type = 'C'
th = 0.1

def make_target(df):
    # 고저점 예측 - 0: 고점 1: 저점
    if target_type == 'P':
        target = []
        for i in range(len(df)):
            if i > len(df) - 1 - pred_term:
                if df.loc[i, '종가'] > df.loc[len(df)-1, '종가']:
                    target.append(0)
                else:
                    target.append(1)
            else:
                if df.loc[i, '종가'] > df.loc[i+pred_term, '종가']:
                    target.append(0)
                else:
                    target.append(1)

        df['target'] = target
    else:
        target = []
        for i in range(len(df)):
            if i > len(df) - 1 - pred_term:
                if 0 > np.average(df.loc[i+1:len(df)-1, '종가'].values.astype(np.float)) - float(df['종가'].values[i]):
                    target.append(0)
                else:
                    target.append(1)
            else:
                if 0 > np.average(df.loc[i+1:i+pred_term-1, '종가'].values.astype(np.float)) - float(df['종가'].values[i]):
                    target.append(0)
                else:
                    target.append(1)

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
