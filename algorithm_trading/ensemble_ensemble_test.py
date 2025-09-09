# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 여러 구간에서 train된 모델들의 앙상블의 performance를 test

import tensorflow as tf
from tensorflow import keras

#from openpyxl import load_workbook

import pandas as pd
import numpy as np
import random

import profit
import ensemble_test as et

pred_term = 20
reinfo_th = 0.5
model_reinfo_th = 0.5
profit.loss_cut = 0.01
profit.profit_cut = 1

import datetime
import sys

remove_columns = ['date']

type = 0 # 0: (0, 1고점, 2저점),  1: (0하락, 1상승)
input_size = 88
n_unit = 200
norm_term = 20

start_time = '2022/03/16/09:00'
end_time = '2022/03/17/15:00'

last_train = '2023-03-15'

et.last_train = last_train
et.start_time = start_time
et.end_time = end_time

import make_reinfo as mr
et.mr = mr
et.tm.mr = mr
pred_term = 20
model_reinfo_th = 0.5
reinfo_th = 0.5
selected_ensemble_types = [0, 5, 7]

selected_num = 3

eval_arr = ''


model = tf.keras.Sequential([
    tf.keras.Input(shape=(input_size)),
    tf.keras.layers.Dense(n_unit, activation='relu'),
    tf.keras.layers.Dense(int(n_unit / 2), activation='relu'),
    tf.keras.layers.Dense(int(n_unit / 4), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy,
              #callbacks=[cp-callback]
              metrics=['accuracy'])


df0_path = 'kospi200f_11_60M.csv'
df_pred_path = last_train+'/kospi200f_60M_pred.csv'
result_path = last_train+'/pred_83_results.csv'

def predict():

    #앙상블들의 예측값 생성
    r = []
    for i in range(selected_num):
        s = eval_arr[selected_ensemble_types[i]][0].strip("[]").replace("\'", "").split(", ")
        et.selected_model_types = s
        #et.selected_model_types = [eval_arr[selected_ensemble_types[i], 0], eval_arr[selected_ensemble_types[i], 1], eval_arr[selected_ensemble_types[i], 2]]
        et.selected_num = len(et.ensembles)
        et.selected_checkpoint_path = [last_train + "/60M_" + et.ensembles[j] + "_best" for j in
                                    range(et.selected_num)]
        et.reinfo_th = reinfo_th
        et.pred_term = pred_term
        et.model_reinfo_th = et.reinfo_th

        et.start_time = start_time
        et.end_time = end_time
        et.last_train = last_train

        et.df_pred_path = df_pred_path
        et.result_path = result_path

        et.predict()

        pred = pd.read_csv(et.result_path, encoding='euc-kr')['result'].values

        r.append(pred)

    r = np.array(r)

    #앙상블의 예측값 생성
    pred = []
    for i in range(len(r[0])):
        cnt = [0, 0, 0]
        for j in range(selected_num):
            cnt[r[j][i]] += 1
        pred.append(np.argmax(cnt))

    # 시가, 고가, 저가, 종가 검색
    df = pd.read_csv(df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= start_time].index.min()
    end_index = df.loc[df['date'] <= end_time].index.max()

    dates = df.pop('date').values[start_index:end_index + 1].reshape(-1)

    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]
    #  0: 고점, 1: 저점

    pred_results = []
    for i in range(len(pred)):
        pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])
    pred_results = np.array(pred_results)

    mr.th = reinfo_th
    mr.pred_term = pred_term
    pred = mr.reinfo(pred, pred_results)
    pred_results[:, 1] = np.array(pred)


    # 결과 파일에 저장
    # 0: 정상, 1: 고점 2:저점
    pd.DataFrame(pred_results, columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(result_path, index=False, encoding='euc-kr')

    # 수익률 계산하여 return
    profit.result_path = result_path
    return profit.calc_profit()

if __name__ == "__main__":

    # type=0 --> 0: 중립 1:고점 2:저점,  type=1 --> n일 후 0:하락  1: 상승
    r = predict()
    print("수익률: " + str(r))

