# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# the functions for self-reflected prediction

import numpy as np
import random
import pandas as pd

pred_term = 5

th = 2
loss_cut = 0.01
profit_cut = 1

trading_9h = False

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

        if trading_9h == True and date[11:13] == '09' and state != 0:
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
            if trading_9h == True:
                state = pred + 1
                count = 1
                buy_price = close
            else:
                state = 0
                count = 0
                buy_price = 0
            continue

        if state == 1:
            if high - buy_price > buy_price * loss_cut:
                profit.append(-int((buy_price*loss_cut+0.05)/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred + 1
                count = 1
                buy_price = close
            elif buy_price -low > buy_price * profit_cut:
                profit.append(int(buy_price*profit_cut/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = 0
                count = 0
                buy_price = close
            #elif count == pred_term - 1:
            #    profit.append((buy_price - close)*250000)
            #    fee.append((buy_price + close)*250000*0.00003)
            #    state = 0
            #    count = 0
            #    buy_price = close
            elif pred == 1:
                profit.append((buy_price - close)*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = 2
                count = 1
                buy_price = close
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                #count %= int(pred_term)
        elif state == 2:
            if buy_price - low > buy_price * loss_cut:
                profit.append(-int((buy_price*loss_cut+0.05)/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = pred + 1
                count = 1
                buy_price = close
            elif high - buy_price > buy_price * profit_cut:
                profit.append(int(buy_price*profit_cut/0.05)*0.05*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = 0
                count = 0
                buy_price = close
            #elif count == pred_term - 1:
            #    profit.append((close - buy_price)*250000)
            #    fee.append((buy_price + close)*250000*0.00003)
            #    state = 0
            #    count = 0
            #    buy_price = close
            elif pred == 0:
                profit.append((close - buy_price)*250000)
                fee.append((buy_price + close)*250000*0.00003)
                state = 1
                count = 1
                buy_price = close
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                #count %= int(pred_term)
        else:
            if pred == 0 or pred == 1:
                state = pred + 1
                count = 1
                buy_price = close
            else:
                count = 0
            profit.append(0)
            fee.append(0)

    profit_rates = (np.array(profit) - np.array(fee)) / (df['close'].values.astype('float') * 2500000 * 0.075 * 1.25) + 1
    return (sum(profit) - sum(fee)), profit_rates

def reinfo(pred, pred_results):
    global th

    pred_results[:, 1] = pred
    df0 = pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close'])

    new_pred = [pred[0], pred[1], pred[2]]
    for i in range(3, len(pred)):

        df = df0[:i+1].reset_index()
        profits, profit_rates = calc_profit(df)
        up = np.count_nonzero(profit_rates > 1)
        down = np.count_nonzero(profit_rates < 1)

        if up * th < down:
            if pred[i] == 1:
                new_pred.append(0)
            elif pred[i] == 0:
                new_pred.append(1)
            else:
                new_pred.append(1)
        else:
            new_pred.append(pred[i])

    return np.array(new_pred)
