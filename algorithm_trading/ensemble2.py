# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 거래에 실재 적용되는 앙상블 모델의 prediction 결과와 수익률

import tensorflow as tf
from tensorflow import keras

#from openpyxl import load_workbook

import pandas as pd
import numpy as np

import datetime
import os

remove_columns = ['date']

input_size = 83
n_unit = 200
norm_term = 20

import profit

profit.loss_cut = 0.01
profit.profit_cut = 1

start_time = '2023/05/16/09:00'
end_time = '2023/05/31/15:00'

model_pools = ['5C', '5HL', '5P', '10C', '10HL', '10P', '15C', '15HL', '15P', '20C', '20HL', '20P',
              '25C', '25HL', '25P', '30C', '30HL', '30P', '40C', '40HL', '40P']

target_type = 'C'

# 수익률을 계산하는 거래 시간
offset = 7

# 2023-11-22 test
last_train = '2023-11-15'
import make_reinfo as mr
pred_term = 10
reinfo_th = 0.3
model_reinfo_th = 0.3
ensembles = ['25C', '30C', '5P']

num_of_models = len(ensembles)
weights = [1 for i in range(num_of_models)]


model = tf.keras.Sequential([
    tf.keras.Input(shape=(input_size)),
    tf.keras.layers.Dense(n_unit, activation='relu'),
    tf.keras.layers.Dense(int(n_unit / 2), activation='relu'),
    tf.keras.layers.Dense(int(n_unit / 4), activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy,
              #callbacks=[cp-callback]
              metrics=['accuracy'])

df0_path = 'kospi200f_11_60M.csv'
df_raw_path = last_train + '/kospi200f_60M_raw.csv'
df_pred_path = last_train + '/kospi200f_60M_pred.csv'
result_path = 'pred_83_results.csv'

def preprocessing():
    df0 = pd.read_csv(df0_path, encoding='euc-kr')
    if os.path.isfile(df_pred_path):
        norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')
        if not norm_df0.empty:

            _start_time = df0.loc[df0['date'] >= start_time].min()['date']
            _end_time = df0.loc[df0['date'] <= end_time].max()['date']

            start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
            last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

            if last_date >= _end_time and start_date <= _start_time:
                print('nothing done! in this preprocessing')
                return

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

    #df0.to_csv(df_raw_path, encoding='euc-kr')

    start_index = df0.loc[df0['date'] >= start_time].index.min()
    end_index = df0.loc[df0['date'] <= end_time].index.max()

    df = df0[start_index - (norm_term-1) : end_index + 1].reset_index(drop=True)
    norm_df = df[norm_term-1 : ].reset_index(drop=True).copy()

    for i in range(end_index - start_index + 1):
        for j in range(1, input_size+1):
            m = df.iloc[i:i+norm_term, j].mean()
            s = df.iloc[i:i+norm_term, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i+norm_term-1, j] - m) / s
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')


def predict():
    # 시가, 고가, 저가, 종가 검색
    df = pd.read_csv(df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= start_time].index.min()
    end_index = df.loc[df['date'] <= end_time].index.max()
    dates = df.pop('date').values[start_index:end_index + 1].reshape(-1)
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]

    dic = {'date': dates, 'result': np.empty(len(open)), 'open': open, 'high': high, 'low': low, 'close': close}
    pred_results = pd.DataFrame(dic).values

    # prediction input, target 생성성
    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')

    start_index = df_pred.loc[df_pred['date'] >= start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    pred_input = df_pred.values[start_index:end_index+1, :input_size].reshape(-1, input_size)

    #모든 모델들의 예측값 생성
    p = ['' for i in range(num_of_models)]
    for i in range(num_of_models):
        p[i] = model_predict(ensembles[i], pred_input, pred_results, model_reinfo_th)

    #weight을 고려한 모델들의 예측값 합성
    pred = []
    for i in range(len(p[0])):
        cnt = [0, 0, 0]
        for j in range(num_of_models):
            cnt[p[j][i]] += weights[j]
        pred.append(np.argmax(cnt))

    # 자기 반성 모드로 prediction조정후 결과 파일에 저장
    dic = {'date': dates, 'result': pred, 'open': open, 'high': high, 'low': low, 'close': close}
    mr.th = reinfo_th
    mr.target_type = target_type
    mr.pred_term = pred_term
    pred = mr.reinfo(pred, pd.DataFrame(dic).values)

    offset_ = min(offset, len(pred))

    dic = {'date': dates, 'result': pred, 'pred1': p[0], 'pred2': p[1], 'pred3': p[2],'open': open, 'high': high, 'low': low, 'close': close}
    pd.DataFrame(dic).to_csv(result_path, index=False, encoding='euc-kr')

    return [dates[-offset_:], pred[-offset_:]]

def model_predict(model_type, pred_input, pred_results, reinfo_th):

    c = model_type.find('C')
    h = model_type.find('HL')
    p = model_type.find('P')

    if c != -1:
        pred_term = int(model_type[:c])
        target_type = 'C'

    elif h != -1:
        pred_term = int(model_type[:h])
        target_type = 'HL'

    elif p != -1:
        pred_term = int(model_type[:p])
        target_type = 'P'

    else:
        print("argument error " + model_type)
        exit(0)

    model.load_weights(last_train + "/60M_" + model_type + "_best")
    pred = model.predict(pred_input)
    pred = np.argmax(pred, axis=1).reshape(-1)

    pred_results[:, 1] = pred

    mr.th = reinfo_th
    mr.pred_term = pred_term
    mr.target_type = target_type
    pred = mr.reinfo(pred, pred_results)

    return pred



if __name__ == "__main__":

    end_time = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    start_time = (datetime.datetime.strptime(last_train, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime("%Y/%m/%d/09:00")

    #end_time = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    #start_time = (datetime.datetime.strptime(end_time, '%Y/%m/%d/%H:%M') - datetime.timedelta(days=15)).strftime("%Y/%m/%d/%H:%M")

    print(df_pred_path)

    print('data processing start...')
    now = datetime.datetime.now()
    print(now)
    preprocessing()
    print('data processing end...')
    r = predict()
    print(r)

    # 수익률 계산하여 return
    profit.result_path = result_path
    p = profit.calc_profit()
    print(p)
