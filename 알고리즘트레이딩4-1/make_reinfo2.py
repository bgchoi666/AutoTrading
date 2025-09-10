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

    return target

def make_train_data_new(df):

    # 고저점 예측 - 0: 정상  1: 고점 2: 저점 until pred_term
    target = []

    for i in range(len(df)):

        if i < pred_term:
            upper = 0
            lower = 0
        else:
            rates = np.array(df[:i]["종가"].rolling(window=pred_term + 1).apply(lambda x: x[pred_term] - x[0]))
            for i in range(pred_term):
                rates[i] = 0

            upper = rates[np.where(rates > 0)].mean() / pred_term
            lower = rates[np.where(rates < 0)].mean() / pred_term

        if i > len(df) - 1 - pred_term:
            if lower > np.average(np.array(df.loc[i+1:len(df)-1, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                target.append(1)
            elif upper < np.average(np.array(df.loc[i+1:len(df)-1, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                target.append(2)
            else:
                target.append(0)
        else:
            if lower > np.average(np.array(df.loc[i+1:i+pred_term, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
                target.append(1)
            elif upper < np.average(np.array(df.loc[i+1:i+pred_term, '종가'].values).astype(np.float) - float(df.loc[i, '종가'])):
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
        n = i - pred_term
        cnt = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for j in range(max(0, i-reinfo_width), i+1):
            if j <= n and pred[j] == target[j]:
                cnt[pred[j]][target[j]] += 1
            if j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) > 0:
                cnt[pred[j]][2] += 1
            elif j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) < 0:
                cnt[pred[j]][1] += 1
            elif j > n and float(pred_results[i, 5]) - float(pred_results[j, 5]) == 0:
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
