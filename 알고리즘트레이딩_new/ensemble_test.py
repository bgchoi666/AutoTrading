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
import make_model as tm

profit.loss_cut = 0.01
profit.profit_cut = 1

import datetime
import sys

remove_columns = ['date']

input_size = 88
n_unit = 200
norm_term = 20

start_time = '2023/11/01/09:00'
end_time = '2023/12/31/15:00'

last_train = '2023-10-31'


import make_reinfo as mr
tm.mr = mr
pred_term = 4
model_reinfo_th = 0.5
reinfo_th = 0.5
selected_model_types = ['5', '6', '8']
#2023-05-30	알고리즘트레이딩_new	6	8	10	0.5	5
#2023-06-01	알고리즘트레이딩_new	5	8	9	0.5	2
#2023-06-02	알고리즘트레이딩_new	5	8	9	0.5	2
#2023-06-19	알고리즘트레이딩_new	6	8	9	0.5	5
#2023-06-27/12:00	알고리즘트레이딩_new	5	6	8	0.33	1
#2023-07-31	알고리즘트레이딩_new	1	4	6	0.4	2




selected_num = len(selected_model_types)
selected_checkpoint_path = [last_train + "/60M_" + selected_model_types[j] + "_best" for j in range(selected_num)]
weights = [1 for i in range(selected_num)]
#weights = [0.34666667, 0.32833333, 0.325]


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
result_path = last_train+'/pred_88_results.csv'

def preprocessing():
    norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')
    df0 = pd.read_csv(df0_path, encoding='euc-kr')

    _end_time = df0.loc[df0['date'] <= end_time].max()['date']
    _start_time = df0.loc[df0['date'] >= start_time].min()['date']

    start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
    last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

    if last_date >= _end_time and start_date <= _start_time:
        print('nothing done! in this preprocessing')
        return

    df0 = pd.read_csv(df0_path, encoding='euc-kr')

    df0["시가대비종가변화율"] = (df0["종가"] - df0["시가"])/df0["시가"]*100
    df0["시가대비고가변화율"] = (df0["고가"] - df0["시가"])/df0["시가"]*100
    df0["시가대비저가변화율"] = (df0["저가"] - df0["시가"])/df0["시가"]*100
    df0["종가대비고가변화율"] = (df0["고가"] - df0["종가"])/df0["종가"]*100
    df0["종가대비저가변화율"] = (df0["저가"] - df0["종가"])/df0["종가"]*100

    df0["1일전"] = np.concatenate([[0], df0["종가"].values[:-1]])
    df0["2일전"] = np.concatenate([[0, 0], df0["종가"].values[:-2]])
    df0["3일전"] = np.concatenate([[0, 0, 0], df0["종가"].values[:-3]])
    df0["4일전"] = np.concatenate([[0, 0, 0, 0], df0["종가"].values[:-4]])
    df0["5일전"] = np.concatenate([[0, 0, 0, 0, 0], df0["종가"].values[:-5]])

    df0["6일전"] = np.concatenate([[0, 0, 0, 0, 0, 0], df0["종가"].values[:-6]])
    df0["7일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-7]])
    df0["8일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-8]])
    df0["9일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-9]])
    df0["10일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-10]])

    df0["1일수익률"] = df0["종가"].rolling(window=2).apply(lambda x: x[1] - x[0])
    df0["3일수익률"] = df0["종가"].rolling(window=4).apply(lambda x: x[3] - x[0])
    df0["5일수익률"] = df0["종가"].rolling(window=6).apply(lambda x: x[5] - x[0])
    df0["10일수익률"] = df0["종가"].rolling(window=11).apply(lambda x: x[10] - x[0])
    df0["20일수익률"] = df0["종가"].rolling(window=21).apply(lambda x: x[20] - x[0])
    df0["40일수익률"] = df0["종가"].rolling(window=41).apply(lambda x: x[40] - x[0])
    df0["60일수익률"] = df0["종가"].rolling(window=61).apply(lambda x: x[60] - x[0])
    df0["90일수익률"] = df0["종가"].rolling(window=91).apply(lambda x: x[90] - x[0])
    df0["120일수익률"] = df0["종가"].rolling(window=121).apply(lambda x: x[120] - x[0])
    df0["180일수익률"] = df0["종가"].rolling(window=181).apply(lambda x: x[180] - x[0])
    df0["240일수익률"] = df0["종가"].rolling(window=241).apply(lambda x: x[240] - x[0])

    df0["5일평균"] = df0["종가"].rolling(window=5).mean()
    df0["20일평균"] = df0["종가"].rolling(window=20).mean()
    df0["60일평균"] = df0["종가"].rolling(window=60).mean()
    df0["120일평균"] = df0["종가"].rolling(window=120).mean()
    df0["240일평균"] = df0["종가"].rolling(window=240).mean()

    df0["5일최고"] = df0["고가"].rolling(window=5).max()
    df0["20일최고"] = df0["고가"].rolling(window=20).max()
    df0["60일최고"] = df0["고가"].rolling(window=60).max()
    df0["120일최고"] = df0["고가"].rolling(window=120).max()
    df0["240일최고"] = df0["고가"].rolling(window=240).max()

    df0["5일최저"] = df0["저가"].rolling(window=5).min()
    df0["20일최저"] = df0["저가"].rolling(window=20).min()
    df0["60일최저"] = df0["저가"].rolling(window=60).min()
    df0["120일최저"] = df0["저가"].rolling(window=120).min()
    df0["240일최저"] = df0["저가"].rolling(window=240).min()

    df0["1일전거래량"] = np.concatenate([[0], df0["거래량"].values[:-1]])
    df0["2일전거래량"] = np.concatenate([[0, 0], df0["거래량"].values[:-2]])
    df0["3일전거래량"] = np.concatenate([[0, 0, 0], df0["거래량"].values[:-3]])
    df0["4일전거래량"] = np.concatenate([[0, 0, 0, 0], df0["거래량"].values[:-4]])
    df0["5일전거래량"] = np.concatenate([[0, 0, 0, 0, 0], df0["거래량"].values[:-5]])

    df0["6일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0], df0["거래량"].values[:-6]])
    df0["7일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-7]])
    df0["8일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-8]])
    df0["9일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-9]])
    df0["10일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-10]])

    df0["1일거래변화량"] = df0["거래량"].rolling(window=2).apply(lambda x: x[1] - x[0])
    df0["3일거래변화량"] = df0["거래량"].rolling(window=4).apply(lambda x: x[3] - x[0])
    df0["5일거래변화량"] = df0["거래량"].rolling(window=6).apply(lambda x: x[5] - x[0])
    df0["10일거래변화량"] = df0["거래량"].rolling(window=11).apply(lambda x: x[10] - x[0])
    df0["20일거래변화량"] = df0["거래량"].rolling(window=21).apply(lambda x: x[20] - x[0])
    df0["40일거래변화량"] = df0["거래량"].rolling(window=41).apply(lambda x: x[40] - x[0])
    df0["60일거래변화량"] = df0["거래량"].rolling(window=61).apply(lambda x: x[60] - x[0])
    df0["90일거래변화량"] = df0["거래량"].rolling(window=91).apply(lambda x: x[90] - x[0])
    df0["120일거래변화량"] = df0["거래량"].rolling(window=121).apply(lambda x: x[120] - x[0])
    df0["180일거래변화량"] = df0["거래량"].rolling(window=181).apply(lambda x: x[180] - x[0])
    df0["240일거래변화량"] = df0["거래량"].rolling(window=241).apply(lambda x: x[240] - x[0])

    df0["5일평균거래량"] = df0["거래량"].rolling(window=5).mean()
    df0["20일평균거래량"] = df0["거래량"].rolling(window=20).mean()
    df0["60일평균거래량"] = df0["거래량"].rolling(window=60).mean()
    df0["120일평균거래량"] = df0["거래량"].rolling(window=120).mean()
    df0["240일평균거래량"] = df0["거래량"].rolling(window=240).mean()

    df0["5일최고거래량"] = df0["거래량"].rolling(window=5).max()
    df0["20일최고거래량"] = df0["거래량"].rolling(window=20).max()
    df0["60일최고거래량"] = df0["거래량"].rolling(window=60).max()
    df0["120일최고거래량"] = df0["거래량"].rolling(window=120).max()
    df0["240일최고거래량"] = df0["거래량"].rolling(window=240).max()

    df0["5일최저거래량"] = df0["거래량"].rolling(window=5).min()
    df0["20일최저거래량"] = df0["거래량"].rolling(window=20).min()
    df0["60일최저거래량"] = df0["거래량"].rolling(window=60).min()
    df0["120일최저거래량"] = df0["거래량"].rolling(window=120).min()
    df0["240일최저거래량"] = df0["거래량"].rolling(window=240).min()

    start_index = df0.loc[df0['date'] >= start_time].index.min()
    end_index = df0.loc[df0['date'] <= end_time].index.max()

    df = df0[start_index - (norm_term-1) : end_index + 1].reset_index(drop=True)
    norm_df = df[norm_term-1 : ].reset_index(drop=True).copy()

    for i in range(end_index - start_index + 1):
        for j in range(1, input_size+1):
            #if j >= 5 and j <= 9:
            #    continue
            m = df.iloc[i:i+norm_term, j].mean()
            s = df.iloc[i:i+norm_term, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i+norm_term-1, j] - m) / s
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')

def predict():
    # create prediction values
    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')

    start_index = df_pred.loc[df_pred['date'] >= start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    pred_input = df_pred.values[start_index:end_index+1, :input_size].reshape(-1, input_size)

    #모델들의 예측값 생성
    r = []
    for i in range(selected_num):
        #print(selected_checkpoint_path[i])
        model.load_weights(selected_checkpoint_path[i])

        tm.reinfo_th = model_reinfo_th
        tm.last_train = last_train
        tm.start_time = start_time
        tm.end_time = end_time
        c = selected_checkpoint_path[i].find('_best')
        if c != -1:
            tm.pred_term = int(selected_checkpoint_path[i][c-1:c])
            if tm.pred_term == 0:
                tm.pred_term = 10
        else:
            print("selected_checkpoint_path error ")
            exit(0)
        tm.df_pred_path = df_pred_path
        tm.result_path = result_path
        pred = tm.predict(model)[:, 1]
        r.append(pred)

    r = np.array(r)

    #앙상블의 예측값 생성
    pred = []
    for i in range(len(r[0])):
        cnt = [0, 0, 0]
        for j in range(selected_num):
            cnt[r[j][i]] += weights[j]
        pred.append(np.argmax(cnt))

    # 시가, 고가, 저가, 종가 검색
    df = pd.read_csv(df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= start_time].index.min()
    end_index = df.loc[df['date'] <= end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]
    #  0: 고점, 1: 저점

    pred_results = []
    for i in range(len(pred)):
        pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])
    pred_results = np.array(pred_results)

    if mr:
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

    print("preprocessing start......")
    preprocessing()
    print("preprocessing end........")

    # type=0 --> 0: 중립 1:고점 2:저점,  type=1 --> n일 후 0:하락  1: 상승
    r = predict()
    print("수익률: " + str(r))

