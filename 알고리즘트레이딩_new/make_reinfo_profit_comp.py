# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# the functions for self-reflected prediction

import numpy as np
import pandas as pd

pred_term = 40
target_type = 'C'

trading_9h = False

th = 1
loss_cut = 0.01

def calc_profit(df):

    state = 0
    count = 0
    buy_price = 0
    profit = []
    fee = []

    for i in range(len(df)):

        pred = int(df.loc[i, 'result'])
        close = float(df.loc[i, 'close'])
        high = float(df.loc[i, 'high'])
        low = float(df.loc[i, 'low'])
        open = float(df.loc[i, 'open'])
        date = df.loc[i, 'date']

        if trading_9h:
            if date[11:13] == '09' and state != 0:
                buy_price = open

        if date[11:13] == '15':
            if state == 1:
                profit.append((buy_price - close) * 250000)
                fee.append((buy_price + close) * 250000 * 0.00003)
            elif state == 2:
                profit.append((close - buy_price) * 250000)
                fee.append((buy_price + close) * 250000 * 0.00003)
            else:
                profit.append(0)
                fee.append(0)
            if trading_9h:
                state = pred + 1
                count = 1
                buy_price = close
            else:
                state = 0
                count = 0
                buy_price = 0
            continue

        if state == 1:

            if high - buy_price >= buy_price * loss_cut:
                profit.append(-int((buy_price*loss_cut+0.05)/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred
                count = 1
                buy_price = close
            elif count == pred_term - 1 or pred == 1:
                profit.append((buy_price - close)*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred + 1
                count = 1
                buy_price = close
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                count %= pred_term
        elif state == 2:

            if buy_price - low >= buy_price * loss_cut:
                profit.append(-int((buy_price*loss_cut+0.05)/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred
                count = 1
                buy_price = close
            elif count == pred_term - 1 or pred == 0:
                profit.append((close - buy_price)*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred + 1
                count = 1
                buy_price = close
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                count %= pred_term
        else:
            if pred == 1 or pred == 0:
                state = pred + 1
                count = 1
                buy_price = close
            else:
                count = 0
            profit.append(0)
            fee.append(0)

    return np.array(profit) - np.array(fee), (np.array(profit) - np.array(fee))/df['close'].values.astype('float').mean()/250000/0.075+1

def reinfo(pred, pred_results):
    global th

    df0 = pd.DataFrame(pred_results, columns=['date', 'result', 'open', 'high', 'low', 'close'])

    new_pred = [pred[0], pred[1], pred[2]]
    for i in range(3, len(pred)):
        df = df0[:i+1].reset_index()
        profits, profit_rates = calc_profit(df)
        profits1 = profits[:int(i*th)+1]
        profits2 = profits[int(i*th):]

        if profits2.sum() < profits1.sum():
            if pred[i] == 1:
                new_pred.append(0)
            else:
                new_pred.append(1)
        else:
            new_pred.append(pred[i])
    return np.array(new_pred)
