# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 개별 모델의 creation, train, prediction

import data

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import gc
import datetime
import random


import make_reinfo as mr


model = ''

def create_model(conf):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(conf.input_size)),
        tf.keras.layers.Dense(conf.n_unit, activation='relu'),
        tf.keras.layers.Dense(int(conf.n_unit / 2), activation='relu'),
        tf.keras.layers.Dense(int(conf.n_unit / 4), activation='relu'),
        tf.keras.layers.Dense(conf.target_num, activation='softmax')
    ])

    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #    checkpoint_path, verbose=1, save_weights_only=True,
        # 다섯 번째 에포크마다 가중치를 저장합니다
    #    save_freq=5)

    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  #callbacks=[cp-callback]
                  metrics=['accuracy'])

    conf.checkpoint_path = conf.last_train+"/60M_input83_test"
    conf.checkpoint_path_best = conf.last_train+"/60M_"+str(conf.pred_term) + conf.target_type + "_best"

    model.save_weights(conf.checkpoint_path)

    return model

def train(model, conf):
    norm_df = pd.read_csv(conf.norm_df_path, encoding='euc-kr')

    train_end_time = datetime.datetime.strptime(conf.last_train, "%Y-%m-%d").strftime("%Y/%m/%d/15:00")
    train_end_index = norm_df.loc[norm_df['date'] <= train_end_time].index.max() - conf.pred_term + 1

    if train_end_index-1000 < 0:
        train_df = norm_df.iloc[:train_end_index]
    else:
        train_df = norm_df.iloc[train_end_index-1000:train_end_index]

    # create train data
    train_df = train_df.drop('date', axis=1, inplace=False)
    train_data = train_df.values

    # downsampling, downsizing the data with target == 0 randomly
    drop_index = []
    r = (len(train_data) / (train_data[:, -1] == 0).sum() - 1) / 2
    for i in range(len(train_data)):
        if train_data[i, -1] == 0 and random.random() > r:
            drop_index.append(i)
    train_data = np.delete(train_data, drop_index, 0)

    pre_accu = 0
    repeat_cnt = 0
    best_profit = 0
    while repeat_cnt < conf.max_repeat_cnt:

        gc.collect()

        repeat_cnt += 1

        # 최근 데이터는 vaidation 데이터로 예약
        train_x = train_data[:, :conf.input_size]
        train_y = train_data[:, conf.input_size]

        # over_sampling
        # X_samp, y_samp = RandomOverSampler(random_state=0).fit_sample(train_x, train_y)
        # X_samp, y_samp = ADASYN(random_state=0, n_neighbors=5).fit_sample(train_x, train_y)
        #train_x, train_y = SMOTETomek(random_state=0).fit_resample(train_x, train_y)


        # train data중 50%만 사용. . .
        #r = random.randrange(1, 10000)
        train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size=conf.train_rate)#, random_state=r)

        # valid data 10% 사용. . .
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)


        # only enough signals are passed
        #n0 = (valid_y == 0).sum()
        #n1 = (valid_y == 1).sum()
        #n2 = (valid_y == 2).sum()
        #if min(n1, n2) < len(valid_y)*0.15:
        #    continue

        # 모델 초기 설정
        try:
            model.load_weights(conf.checkpoint_path_best)
        except:
            model.load_weights(conf.checkpoint_path)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, verbose=0)
        model.fit(train_x, train_y, batch_size=conf.batch_size, epochs=conf.epochs, verbose=0, callbacks=[early_stopping],
                  validation_data=(valid_x, valid_y))

        prediction = model.predict(valid_x).reshape(-1, conf.target_num)
        prediction = np.argmax(prediction, axis=1).reshape(-1)

        # calculate accuracy
        c = 0
        s = list(prediction).count(0) + list(prediction).count(1)
        for i in range(len(prediction)):
            #if valid_y[i] == 0. and prediction[i] == 0.:
            #        c += 1
            if valid_y[i] == 1. and prediction[i] == 1.:
                    c += 1
            elif valid_y[i] == 0. and prediction[i] == 0.:
                    c += 1
        if s == 0:
            accu = 0
        else:
            accu = c / s

        if repeat_cnt % 100 == 0:
            print("반복 횟수 : " + str(repeat_cnt) + " accuracy = " + str(accu))

        if accu > pre_accu:
            best_x = train_x
            best_y = train_y
            # model.fit(valid_x, valid_y, batch_size=batch_size, epochs=3, verbose=0)
            model.save_weights(conf.checkpoint_path_best)

            print("best accuracy " + str(accu))
            pre_accu = accu

        if pre_accu >= 1.0:
            break

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

def test(model, conf):
    norm_df = pd.read_csv(conf.df_pred_path, encoding='euc-kr')
    #train_end_index = norm_df.loc[norm_df['date'] <= start_time].index.max()
    start_index = norm_df.loc[norm_df['date'] >= conf.start_time].index.min()
    end_index = norm_df.loc[norm_df['date'] <= conf.end_time].index.max()

    norm_df.drop('date', axis=1, inplace=True)

    # create test data
    test_data = norm_df.iloc[start_index:end_index+1].values
    test_x = test_data[:, :conf.input_size]
    test_y = test_data[:, conf.input_size]

    # 선택 2: 저장된 최고의 학습 weight을 reload하여 곧바로 예측
    #model.load_weights(checkpoint_path_best)

    test_prediction = model.predict(test_x)
    test_prediction = np.argmax(test_prediction, axis=1).reshape(-1)

    # calculate accuracy
    c = 0
    s = list(test_prediction).count(0) + list(test_prediction).count(1)#len(test_prediction)
    for i in range(len(test_prediction)):
        #if test_y[i] == 0 and test_prediction[i] == 0:
        #    c += 1
        #if test_y[i] == 0 and test_prediction[i] == 0:
        #    c += 1
        if test_y[i] == 1 and test_prediction[i] == 1:
            c += 1
        elif test_y[i] == 0 and test_prediction[i] == 0:
            c += 1
    print('accuracy = ', c / s)

def predict(model, conf):

    df_pred = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

    if conf.start_time > df_pred['date'].values[-1]:
        conf.start_time = df_pred['date'].values[-1]

    start_index = df_pred.loc[df_pred['date'] >= conf.start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= conf.end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    pred_input = df_pred.values[start_index:end_index+1, :conf.input_size].reshape(-1, conf.input_size)

    #model.load_weights(checkpoint_path_best)
    pred = model.predict(pred_input)
    pred = np.argmax(pred, axis=1).reshape(-1)

    # 종가 검색
    df = pd.read_csv(conf.df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= conf.start_time].index.min()
    end_index = df.loc[df['date'] <= conf.end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]

    #  0: 고점, 1: 저점
    pred_results = [dates, pred, open, high, low, close]
    pred_results = np.array(pred_results).transpose()

    if conf.reinfo_th > 0:
        mr.th = conf.reinfo_th
        mr.pred_term = conf.pred_term
        mr.target_type = conf.target_type
        pred = mr.reinfo(pred, pred_results, conf.start_time, conf.reinfo_width)
        pred_results[:, 1] = pred

    # 앙상불 예측(ensemble.py)의 경우 개별 모델들의 result_path에 last_train 폴더를 붙이기 위해서.
    # 이는 자동 거래시 앙상블 예측의 결과가 끝나기 전에 모델의 결과를 앙상블의 결과로 읽는 오류를 막기 위해서이다.
    if conf.result_path.find(conf.last_train) < 0:
        result_path = conf.last_train + "/" + conf.result_path
    else:
        result_path = conf.result_path

    pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(result_path, index=False, encoding='euc-kr')
    return pred_results[len(pred_results)-len(dates):]

def parse(type):

    # 종가, 고가 기준에 따라 target_prob0, chkpoint_best file path 조정
    c = type.find('C')
    h = type.find('HL')
    p = type.find('P')
    if c != -1:
        pred_term = int(type[:c])
        target_type = 'C'
        base1 = '종가'
        base2 = '종가'
    elif h != -1:
        pred_term = int(type[:h])
        target_type = 'HL'
        base1 = '고가'
        base2 = '저가'
    elif p != -1:
        pred_term = int(type[:p])
        target_type = 'P'
        base1 = '종가'
        base2 = '종가'
    else:
        print("argument error " + type)
        exit(0)

    return pred_term, target_type, base1, base2
