# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# the functions for self-reflected prediction

import numpy as np
import random
import pandas as pd

pred_term = 40
target_type = 'C'

th = 0.5

def make_train_data(df):

    # 고저점 예측 - 0: 정상  1: 고점 2: 저점 until pred_term
    target = []
    if target_type == 'P':
        for i in range(len(df)):
            if i > len(df) - 1 - pred_term:
                if df.loc[i, '종가'] > df.loc[len(df) - 1, '종가']:
                    target.append(1)
                elif df.loc[i, '종가'] < df.loc[len(df) - 1, '종가']:
                    target.append(2)
                else:
                    target.append(0)
            else:
                if df.loc[i, '종가'] > df.loc[i+pred_term, '종가']:
                    target.append(1)
                elif df.loc[i, '종가'] < df.loc[i+pred_term, '종가']:
                    target.append(2)
                else:
                    target.append(0)
    elif target_type == 'C':
        for i in range(len(df)):
            if i > len(df) - 1 - pred_term:
                if 0 > np.average(np.array(df.loc[i+1:len(df)-1, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(1)
                elif 0 < np.average(np.array(df.loc[i+1:len(df)-1, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(2)
                else:
                    target.append(0)
            else:
                if 0 > np.average(np.array(df.loc[i+1:i+pred_term, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(1)
                elif 0 < np.average(np.array(df.loc[i+1:i+pred_term, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(2)
                else:
                    target.append(0)
    elif target_type == 'HL':
        for i in range(len(df)):
            if i > len(df) - 1 - pred_term:
                if 0 > np.average(np.array(df.loc[i+1:len(df)-1, '고가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(1)
                elif 0 < np.average(np.array(df.loc[i+1:len(df)-1, '저가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(2)
                else:
                    target.append(0)
            else:
                if 0 > np.average(np.array(df.loc[i+1:i+pred_term, '고가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(1)
                elif 0 < np.average(np.array(df.loc[i+1:i+pred_term, '저가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                    target.append(2)
                else:
                    target.append(0)
    else:
        target = [0 for k in range(len(df))]

    return target

def reinfo(pred, pred_results, start_time, reinfo_width):
    global th

    df = pd.DataFrame(pred_results, columns=['date', 'results', '시가', '고가', '저가', '종가'])
    target = make_train_data(df)
    start_index = df.loc[df['date'] >= start_time].index.min()

    new_pred = []
    for i in range(start_index, len(pred)):
        n = i - pred_term
        cnt = 0
        for j in range(max(0, i-reinfo_width), i+1):
            if j <= n and pred[j] == target[j]:
                cnt += 1
            if j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) > 0:
                if pred[i] == 2:
                    cnt += 1
            elif j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) < 0:
                if pred[i] == 1:
                    cnt += 1
            elif j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) == 0:
                if pred[i] == 0:
                    cnt += 1

        prob = cnt / (min(i, reinfo_width) + 0.00001)

        #if th == 'r':
        #    th = random.random()
        if prob < th:
            if pred[i] == 1:
                new_pred.append(2)
            elif pred[i] == 2:
                new_pred.append(1)
            else:
                new_pred.append(0)
        else:
            new_pred.append(pred[i])
    return np.array(new_pred)
