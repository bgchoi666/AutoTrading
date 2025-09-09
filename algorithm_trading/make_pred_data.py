# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

import pandas as pd
import sys

input_size = 83

future_day = 5

start_time = '2021/04/01/09:00'
end_time = '2021/08/25/14:00'

raw_df_path = 'test/kospi200f_60M_raw.csv'
df_pred_path = 'test/kospi200f_60M_pred.csv'

df0 = pd.read_csv(raw_df_path, encoding='euc-kr')
norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')

start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

if last_date >= end_time and start_date <= start_time:
    print('nothing done!')
    sys.exit()

start_index = df0.loc[df0['date'] >= start_time].index.min()
end_index = df0.loc[df0['date'] <= end_time].index.max()

df = df0[start_index - 19: end_index + 1].reset_index(drop=True)
norm_df = df[19:].reset_index(drop=True).copy()

for i in range(end_index - start_index + 1):
    for j in range(1, input_size + 1):
        m = df.iloc[i:i + 20, j].mean()
        s = df.iloc[i:i + 20, j].std()
        if s == 0:
            norm_df.iloc[i, j] = 0
        else:
            norm_df.iloc[i, j] = (df.iloc[i + 19, j] - m) / s
norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')
