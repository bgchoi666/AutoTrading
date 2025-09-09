# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# the functions for self-reflected prediction

import numpy as np
import random
import pandas as pd

pred_term = 5

th = 0.5
loss_cut = 0.01
profit_cut = 1

trading_9h = False

def calc_profit(df):

    state = 0
    buy_price = 0
    profit = []

    for i in range(len(df)):

        pred = int(df.loc[i, 'result'])
        close = float(df.loc[i, 'close'])
        high = float(df.loc[i, 'high'])
        low = float(df.loc[i, 'low'])
        open = float(df.loc[i, 'open'])
        date = df.loc[i, 'date']

        if trading_9h == True and date[11:13] == '09' and state != 0:
            buy_price = open

        if date[11:13] == '15' or i == len(df) - 1:
            if state == 1:
                profit.append(np.sign(buy_price - close))
            elif state == 2:
                profit.append(np.sign(close - buy_price))
            else:
                profit.append(0)
            if trading_9h == True:
                state = pred + 1
                buy_price = close
            else:
                state = 0
                buy_price = 0
            continue

        if state == 1:
            if high - buy_price > buy_price * loss_cut:
                profit.append(-1)
                state = pred + 1
                buy_price = close
            elif buy_price -low > buy_price * profit_cut:
                profit.append(1)
                state = 0
                buy_price = close
            elif pred == 1:
                profit.append(np.sign(buy_price - close))
                state = 2
                buy_price = close
            else:
                profit.append(0)
        elif state == 2:
            if buy_price - low > buy_price * loss_cut:
                profit.append(-1)
                state = pred + 1
                buy_price = close
            elif high - buy_price > buy_price * profit_cut:
                profit.append(1)
                state = 0
                buy_price = close
            elif pred == 0:
                profit.append(np.sign(close - buy_price))
                state = 1
                buy_price = close
            else:
                profit.append(0)
        else:
            state = pred + 1
            buy_price = close
            profit.append(0)

    profit_pos = [n for n in profit if n > 0]
    profit_neg = [n for n in profit if n < 0]

    if len(profit_pos) +len(profit_neg) == 0:
        return 1
    else:
        return len(profit_pos) / (len(profit_pos) + len(profit_neg))

def reinfo(pred, pred_results):
    global th

    pred_results[:, 1] = pred
    df0 = pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close'])

    new_pred = [pred[0], pred[1], pred[2]]
    for i in range(3, len(pred)):

        df = df0[:i+1].reset_index()
        rate = calc_profit(df)

        if rate < th:
            if pred[i] == 1:
                new_pred.append(0)
            elif pred[i] == 0:
                new_pred.append(1)
            else:
                new_pred.append(1)
        else:
            new_pred.append(pred[i])

    return np.array(new_pred)
