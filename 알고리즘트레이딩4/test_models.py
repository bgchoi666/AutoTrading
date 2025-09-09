# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기존 해당 기간의 trained 모델들의 주어진 기간에서의 수익률 비교 파일 생성

import pandas as pd
import numpy as np
import make_model as ep
import profit
import sys
import os
from datetime import datetime

last_train = '2022-05-31'
start_time = '2022/06/01/09:00'
end_time = '2022/06/15/15:00'

models = ['', "5C", "5HL", "5P", "10C", "10HL", "10P", "15C", "15HL", "15P", "20C", "20HL", "20P",
          "25C", "25HL", "25P", "30C", "30HL", "30P", "40C", "40HL", "40P"]

file_name = last_train+"/models.csv"

df0_path = 'kospi200f_11_60M.csv'
df_pred_path = last_train+'/kospi200f_60M_pred.csv'
result_path = last_train+'/pred_83_results.csv'

def preprocessing():
    norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')
    df0 = pd.read_csv(df0_path, encoding='euc-kr')

    _end_time = df0.loc[df0['date'] <= end_time].max()['date']

    start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
    last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

    if last_date >= _end_time and start_date <= start_time:
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

    df = df0[start_index - (ep.norm_term-1) : end_index + 1].reset_index(drop=True)
    norm_df = df[ep.norm_term-1 : ].reset_index(drop=True).copy()

    for i in range(end_index - start_index + 1):
        for j in range(1, ep.input_size+1):
            m = df.iloc[i:i+ep.norm_term, j].mean()
            s = df.iloc[i:i+ep.norm_term, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i+ep.norm_term-1, j] - m) / s
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')

def run():
    """
    # 전처리
    preprocessing()

    # input data 생성
    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')
    start_index = df_pred.loc[df_pred['date'] >= start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= end_time].index.max()
    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)
    pred_input = df_pred.values[start_index:end_index + 1, :ep.input_size].reshape(-1, ep.input_size)

    # 시가, 고가, 저가, 종가 데이터 save
    df = pd.read_csv(df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= start_time].index.min()
    end_index = df.loc[df['date'] <= end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]
    """
    rates = []
    for i in range(1, 22):
        # model setup
        # 종가, 고가 기준에 따라 target_prob0, chkpoint_best file path 조정
        c = models[i].find('C')
        h = models[i].find('HL')
        p = models[i].find('P')
        if c != -1:
            ep.pred_term = int(models[i][:c])
            ep.target_type = 'C'
            ep.base1 = '종가'
            ep.base2 = '종가'
        elif h != -1:
            ep.pred_term = int(models[i][:h])
            ep.target_type = 'HL'
            ep.base1 = '고가'
            ep.base2 = '저가'
        elif p != -1:
            ep.pred_term = int(models[i][:p])
            ep.target_type = 'P'
            ep.base1 = '종가'
            ep.base2 = '종가'
        else:
            print("argument error " + models[i])
            exit(0)

        """
        ep.checkpoint_path = last_train + "/60M_input83_test"

        model = ep.create_model(3)
        model.load_weights(checkpoint_path_best)
        pred = model.predict(pred_input)
        pred = np.argmax(pred, axis=1).reshape(-1)

        #  0: 정상, 1: 매도 2:매수
        pred_results = []
        for i in range(len(dates)):
            pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])
        pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(
            result_path, index=False, encoding='euc-kr')

        profit.pred_term = ep.pred_term
        profit.result_path = result_path
        p = profit.calc_profit()
        print(checkpoint_path_best + " 수익률: " + str(p))
        """

        ep.last_train = last_train
        ep.start_time = start_time
        ep.end_time = end_time

        ep.checkpoint_path = ep.last_train + "/60M_input83_test"
        ep.checkpoint_path_best = ep.last_train + "/60M_" + models[i] + "_best"

        ep.df_raw_path = ep.last_train + '/kospi200f_60M_raw.csv'
        ep.norm_df_path = ep.last_train + '/kospi200f_60M_norm.csv'
        ep.df_pred_path = ep.last_train + '/kospi200f_60M_pred.csv'
        ep.result_path = ep.last_train + '/pred_83_results.csv'

        ep.model = ep.create_model(ep.target_num)

        print('data processing start...')
        now = datetime.now()
        print(now)
        ep.gubun = 0
        ep.preprocessing()
        print('data processing end...')

        ep.reinfo_no = ""
        r = ep.predict(ep.model)

        profit.loss_cut = 1
        profit.profit_cut = 1
        profit.pred_term = ep.pred_term
        profit.result_path = ep.result_path
        p = max(profit.calc_profit(), 0.5)

        print(ep.last_train + " " + models[i] + " 수익률: " + str(p))

        rates.append(p)
    if not os.path.isfile(file_name):
        pd.DataFrame({'model': models[1:], start_time[5:10] + "~" + end_time[5:10]: rates}).to_csv(file_name, index=False, encoding='euc-kr')
    else:
        df = pd.read_csv(file_name, encoding="euc-kr")
        df[start_time[5:10] + "~" + end_time[5:10]] = rates
        df.to_csv(file_name, index=False, encoding="euc-kr")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        last_train = sys.argv[1]
        start_time = sys.argv[2]
        end_time = sys.argv[3]

        file_name = last_train + "/models.csv"
        df_pred_path = last_train + '/kospi200f_60M_pred.csv'
        result_path = last_train + '/pred_83_results.csv'
    run()
    sys.exit(0)