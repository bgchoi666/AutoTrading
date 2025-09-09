# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 모델 타입을 생성하여 주어진 구간에서 train, test, prediction
# - 현재 종가가 다음 n일 동안 최고가 이면 0, 최저가 이면 1
# - 중립 state는 없음
# DMI, stocastic은 non-normalization

import tensorflow as tf
from tensorflow import keras

import make_reinfo as mr
reinfo_th = 0.1
loss_cut = 1
profit_cut = 1

from openpyxl import load_workbook

import pandas as pd
import numpy as np
import random
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math
import datetime
import random
import sys
import os

reinfo_no = ""

#from imblearn.under_sampling import *
#from imblearn.over_sampling import *
#from imblearn.combine import *

remove_columns = ['date']

input_size = 88
n_unit = 200
batch_size = 20
epochs=30
train_size = 0.9
train_offset = 240
gubun = 2 # 0:predict only 1:test only 2:train
max_repeat_cnt = 100

pred_term = 5
norm_term = 20
target_type = 'HL'
train_rate = 0.5
base1 = '고가'
base2 = '저가'
target_num = 2

last_train = '2017-12-31'
start_time = '2018/01/01/09:00'
end_time = '2018/01/15/15:00'


checkpoint_path = last_train+"/60M_input83_test"
checkpoint_path_best = last_train+"/60M_"+str(pred_term) + target_type + "_best"

model = ''

def create_model(target_num):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_size)),
        tf.keras.layers.Dense(n_unit, activation='relu'),
        tf.keras.layers.Dense(int(n_unit / 2), activation='relu'),
        tf.keras.layers.Dense(int(n_unit / 4), activation='relu'),
        tf.keras.layers.Dense(target_num, activation='softmax')
    ])

    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #    checkpoint_path, verbose=1, save_weights_only=True,
        # 다섯 번째 에포크마다 가중치를 저장합니다
    #    save_freq=5)

    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  #callbacks=[cp-callback]
                  metrics=['accuracy'])
    model.save_weights(checkpoint_path)

    return model

df0_path = 'kospi200f_11_60M.csv'
df_raw_path = last_train+'/kospi200f_60M_raw.csv'
norm_df_path = last_train+'/kospi200f_60M_norm.csv'
df_pred_path = last_train+'/kospi200f_60M_pred.csv'
result_path = last_train+'/pred_88_results.csv'


# *_pred.csv 파일이 존재할 떄 undersampling 비율 (target_prob*)에 따라 _norm.csv 파일 생성
def make_train_data():
    df = pd.read_csv(df_raw_path, encoding='euc-kr')
    norm_df = pd.read_csv(df_pred_path, encoding='euc-kr')

    # 고저점 예측 - 0: 정상  1: 고점 2: 저점 until pred_term, 단순 예측 - 0: 하락 1: 상승
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

        norm_df['target'] = target[19:]
    else:
        target = []
        for i in range(len(df)):
            if i > len(df) - 1 - pred_term:
                if 0 > np.average(df.loc[i+1:len(df)-1, base1].values - df.loc[i, '종가']):
                    target.append(0)
                else:
                    target.append(1)
            else:
                if 0 > np.average(df.loc[i+1:i+pred_term-1, base1].values - df.loc[i, '종가']):
                    target.append(0)
                else:
                    target.append(1)
        norm_df['target'] = target[19:]

    norm_df.to_csv(norm_df_path, index=False, encoding='euc-kr')
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')

    print('train을 위한 _norm.csv 파일 생성 완료')


# 사실상 사용안함 , preprocessing은 'make_raw_data.py'에 의해 파생변수 생성하고 normalization은 수작업을 통해 *_pred,csv 생성
def preprocessing():
    # 필요 구간의 전처리 데이터 존재여부에 따라 처리

    if not os.path.isfile(df_pred_path):
        print("==============================================")
    else:
        norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')

        df0 = pd.read_csv(df0_path, encoding='euc-kr')
        _end_time = df0.loc[df0['date'] <= end_time].max()['date']
        _start_time = df0.loc[df0['date'] >= start_time].min()['date']
        start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
        last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

        if gubun == 0 or gubun == 1: # for prediction
            if last_date >= _end_time and start_date <= _start_time:
                print('nothing done! in this preprocessing')
                return
        else:
            train_start_index = norm_df0.loc[norm_df0['date'] >= start_time].index.min() - 1000
            if last_date >= _end_time and train_start_index > 0:
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

    if gubun == 0:
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
        return

    train_start_index = df0.loc[df0['date'] >= start_time].index.min() - 1020
    if train_start_index < 19:
        train_start_index = 19
    train_end_index = df0.loc[df0['date'] <= end_time].index.max()
    df = df0[train_start_index:train_end_index+1].reset_index(drop=True)
    df.to_csv(df_raw_path, encoding='euc-kr')

    norm_df = df.copy()
    for i in range(norm_term-1, len(norm_df)):
        for j in range(1, input_size+1):
            #if j >=5 and j <= 9:
            #    continue
            m = df.iloc[i-(norm_term-1):i+1, j].mean()
            s = df.iloc[i - (norm_term-1):i+1, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i, j] - m) / s
    norm_df = norm_df.loc[norm_term-1:].reset_index(drop=True)

    # save the normalized data for prediction
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')

def train(model, num):
    import profit

    norm_df = pd.read_csv(norm_df_path, encoding='euc-kr')

    train_end_time = datetime.datetime.strptime(last_train, "%Y-%m-%d").strftime("%Y/%m/%d/15:00")
    train_end_index = norm_df.loc[norm_df['date'] <= train_end_time].index.max() - pred_term + 1

    #pred_end_index = norm_df.loc[norm_df['date'] <= start_time].index.max()
    #pred_start_date = norm_df.loc[pred_end_index - 35, 'date']
    #pred_end_date = norm_df.loc[pred_end_index, 'date']

    if train_end_index-1000 < 0:
        train_df = norm_df.iloc[:train_end_index]
    else:
        train_df = norm_df.iloc[train_end_index-1000:train_end_index]#.drop(list(valid_df.index), axis=0)

    # create train data
    train_df = train_df.drop('date', axis=1, inplace=False)
    train_data = train_df.values

    # downsampling, downsizing the data with target == 0 randomly
    #drop_index = []
    #r = (len(train_data) / (train_data[:, -1] == 0).sum() - 1) / 2
    #for i in range(len(train_data)):
    #    if train_data[i, -1] == 0 and random.random() > r:
    #        drop_index.append(i)
    #train_data = np.delete(train_data, drop_index, 0)

    pre_accu = 0
    repeat_cnt = 0
    best_profit = 0
    while repeat_cnt < max_repeat_cnt:

        gc.collect()

        repeat_cnt += 1

        # 최근 데이터는 vaidation 데이터로 예약
        train_x = train_data[:, :input_size]
        train_y = train_data[:, input_size]

        # over_sampling
        # X_samp, y_samp = RandomOverSampler(random_state=0).fit_sample(train_x, train_y)
        # X_samp, y_samp = ADASYN(random_state=0, n_neighbors=5).fit_sample(train_x, train_y)
        #train_x, train_y = SMOTETomek(random_state=0).fit_resample(train_x, train_y)


        # train data중 50%만 사용. . .
        #r = random.randrange(1, 10000)
        train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size=train_rate)#, random_state=r)

        # valid data 10% 사용. . .
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)


        # target 재설정하여 model training
        model.load_weights(checkpoint_path)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, verbose=0)
        model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[early_stopping],
                  validation_data=(valid_x, valid_y))

        prediction = model.predict(valid_x).reshape(-1, num)
        prediction = np.argmax(prediction, axis=1).reshape(-1)

        # calculate accuracy
        c = 0
        s = 0  # len(prediction)
        for i in range(len(prediction)):
            # if valid_y[i] == 0.:
            #    if prediction[i] == 0.:
            #        c += 1
            if valid_y[i] == 0.:
                s += 1
                if prediction[i] == 0.:
                    c += 1
            elif valid_y[i] == 1.:
                s += 1
                if prediction[i] == 1.:
                    c += 1
        accu = c / s
        if repeat_cnt % 100 == 0:
            print("반복 횟수 : " + str(repeat_cnt) + " accuracy = " + str(accu))

        if accu > pre_accu:
            best_x = train_x
            best_y = train_y
            # model.fit(valid_x, valid_y, batch_size=batch_size, epochs=3, verbose=0)
            model.save_weights(checkpoint_path_best)

            print("best accuracy " + str(accu))
            pre_accu = accu
        """
        r = predict(model)
        profit.pred_term = pred_term
        profit.result_path = result_path
        p = profit.calc_profit()

        # save the best moodel
        if float(best_profit) < float(p):
            model.save_weights(checkpoint_path_best)
            best_profit = p
            print(str(sys.argv[2]) + " 수익률: " + str(p))
        """
    print("best accuracy " + str(pre_accu))

def test(model):
    norm_df = pd.read_csv(df_pred_path, encoding='euc-kr')
    #train_end_index = norm_df.loc[norm_df['date'] <= start_time].index.max()
    start_index = norm_df.loc[norm_df['date'] >= start_time].index.min()
    end_index = norm_df.loc[norm_df['date'] <= end_time].index.max()

    norm_df.drop('date', axis=1, inplace=True)

    # create test data
    test_data = norm_df.iloc[start_index:end_index+1].values
    test_x = test_data[:, :input_size]
    test_y = test_data[:, input_size]

    # 선택 2: 저장된 최고의 학습 weight을 reload하여 곧바로 예측
    #model.load_weights(checkpoint_path_best)

    test_prediction = model.predict(test_x)
    test_prediction = np.argmax(test_prediction, axis=1).reshape(-1)

    # calculate accuracy
    c = 0
    s = len(test_prediction)
    for i in range(len(test_prediction)):
        #if test_y[i] == 0 and test_prediction[i] == 0:
        #    c += 1
        if test_y[i] == 0 and test_prediction[i] == 0:
            c += 1
        elif test_y[i] == 1 and test_prediction[i] == 1:
            c += 1
    print('accuracy = ', c / s)

def predict(model):
    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')

    start_index = df_pred.loc[df_pred['date'] >= start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    pred_input = df_pred.values[start_index:end_index+1, :input_size].reshape(-1, input_size)

    #model.load_weights(checkpoint_path_best)
    pred = model.predict(pred_input)
    pred = np.argmax(pred, axis=1).reshape(-1)

    # 종가 검색
    df = pd.read_csv(df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= start_time].index.min()
    end_index = df.loc[df['date'] <= end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]

    #  0: 고점, 1: 저점
    pred_results = [dates, pred, open, high, low, close]
    pred_results = np.array(pred_results).transpose()

    mr.th = reinfo_th
    mr.pred_term = pred_term
    mr.target_type = target_type
    pred = mr.reinfo(pred, pred_results)
    pred_results[:, 1] = pred

    #for i in range(len(dates)):
    #    #pred_results = pred_results.append({'date': dates[i], 'result': pred[i]}, ignore_index=True)
    #    pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])


    #pred_results.to_csv(result_path, index=False, encoding='euc-kr')
    pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(result_path, index=False, encoding='euc-kr')
    return pred_results[len(pred_results)-len(dates):]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        gubun = 0
    else:
        gubun = int(sys.argv[1])
        # 종가, 고가 기준에 따라 target_prob0, chkpoint_best file path 조정
        c = sys.argv[2].find('C')
        h = sys.argv[2].find('HL')
        p = sys.argv[2].find('P')
        if c != -1:
            pred_term = int(sys.argv[2][:c])
            target_type = 'C'
            base1 = '종가'
            base2 = '종가'
        elif h != -1:
            pred_term = int(sys.argv[2][:h])
            target_type = 'HL'
            base1 = '고가'
            base2 = '저가'
        elif p != -1:
            pred_term = int(sys.argv[2][:p])
            target_type = 'P'
            base1 = '종가'
            base2 = '종가'
        else:
            print("argument error " + sys.argv[2])
            exit(0)

        if len(sys.argv) >= 6:

            last_train = sys.argv[3]
            start_time = sys.argv[4]
            end_time = sys.argv[5]

            checkpoint_path = last_train + "/60M_input88_test"
            checkpoint_path_best = last_train + "/60M_" + str(pred_term) + target_type + "_best"

            df_raw_path = last_train + '/kospi200f_60M_raw.csv'
            norm_df_path = last_train + '/kospi200f_60M_norm.csv'
            df_pred_path = last_train + '/kospi200f_60M_pred.csv'
            result_path = last_train + '/pred_88_results.csv'

        else:
            print('training, test date error!')
            exit(1)

    model = create_model(target_num)

    if gubun == 0:
        print('data processing start...')
        now = datetime.datetime.now()
        print(now)
        preprocessing()
        print('data processing end...')

        model.load_weights(checkpoint_path_best)
        r = predict(model)

        import profit
        profit.pred_term = pred_term
        profit.result_path = result_path
        p = profit.calc_profit()
        print(sys.argv[2] + " 수익률: " + str(p))

    elif gubun == 1:
        now = datetime.datetime.now()
        print(now)
        model.load_weights(checkpoint_path_best)
        test(model)

    else:
        print('data processing start...')
        now = datetime.datetime.now()
        print(now)
        preprocessing()
        print('data processing end...')

        make_train_data()

        now = datetime.datetime.now()
        print(now)
        train(model, target_num)
        print('training end...')

        now = datetime.datetime.now()
        print(now)
        test(model)

        r = predict(model)
        #print(r)

        import profit
        profit.loss_cut = loss_cut
        profit.profit_cut = profit_cut
        profit.pred_term = pred_term
        profit.result_path = result_path
        p = profit.calc_profit()
        print(str(sys.argv[2]) + " 수익률: " + str(p))