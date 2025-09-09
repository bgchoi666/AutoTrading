# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

input_size = 83
train_offset = 240

future_day = 5
type = 2 # 0: x일후 상승 하락  1: x일 전후 종가 기준 고점, 저점  2: x일후 고점(고가 대비) 저점(저가대비)

df0_path = 'kospi200f_11_60M.csv'
raw_df_path = 'test/kospi200f_60M_raw.csv'


df0 = pd.read_csv(df0_path, encoding='euc-kr')

df0["시가대비종가변화율"] = (df0["종가"] - df0["시가"]) / df0["시가"] * 100
df0["시가대비고가변화율"] = (df0["고가"] - df0["시가"]) / df0["시가"] * 100
df0["시가대비저가변화율"] = (df0["저가"] - df0["시가"]) / df0["시가"] * 100
df0["종가대비고가변화율"] = (df0["고가"] - df0["종가"]) / df0["종가"] * 100
df0["종가대비저가변화율"] = (df0["저가"] - df0["종가"]) / df0["종가"] * 100

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


df = df0[train_offset:].reset_index(drop=True)

if type == 2:
    # target column 생성 : 0: 중립  1:고점, 2:저점  고가, 저가 대비
    고저점 = []  # [0 for i in range(future_day-1)]

    for i in range(len(df)):
        if i > len(df) - future_day:
            고저점.append(0)
        else:
            if df.loc[i, '종가'] >= max(df.loc[i + 1:i + future_day - 1, '고가']):
                고저점.append(1)
            elif df.loc[i, '종가'] <= min(df.loc[i + 1:i + future_day - 1, '저가']):
                고저점.append(2)
            else:
                고저점.append(0)
    df['target'] = 고저점
elif type == 1:
    # target column 생성 : 0: 중립  1:고점, 2:저점   종가 대비
    고저점 = [0 for i in range(future_day - 1)]

    for i in range(future_day-1, len(df)):
        if i > len(df) - future_day:
            고저점.append(0)
        else:
            if df.loc[i, '종가'] >= max(df.loc[i - future_day + 1:i + future_day - 1, '종가']):
                고저점.append(1)
            elif df.loc[i, '종가'] <= min(df.loc[i - future_day + 1:i + future_day - 1, '종가']):
                고저점.append(2)
            else:
                고저점.append(0)
    df['target'] = 고저점
else:
    target = []
    for i in range(len(df)):
        if i > len(df) - 1 - future_day:
            target.append(0)
        else:
            if df.loc[i, '종가'] > df.loc[i+future_day, '종가']:
                target.append(1)
            else:
                target.append(0)
    df['target'] = target

df.to_csv(raw_df_path, index=False, encoding='euc-kr')