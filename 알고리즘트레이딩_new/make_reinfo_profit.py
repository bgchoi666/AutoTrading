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

    # 15시 종가를 익일 시가로 조정
    #for i in range(len(df)-1):
    #    if i != 0 and df.loc[i, 'date'][11:13] == '15':
    #        df.loc[i, 'close'] = df.loc[i+1, 'open']

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
        date = df.loc[i, 'date']
        open = df.loc[i, 'open']

        #if buy_price == 'end' and not date[11:13] == '09':
        #    profit.append(0)
        #    fee.append(0)
        #    continue

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
                count %= pred_term
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
                count %= pred_term
        else:
            if pred == 0 or pred == 1:
                state = pred + 1
                count = 1
                buy_price = close
            else:
                count = 0
            profit.append(0)
            fee.append(0)

    a = sum(profit) - sum(fee)
    b = df['close'].values.astype(np.float).mean()

    return a / (b*1.25*250000*0.075) + 1

def reinfo(pred, pred_results):
    global th

    pred_results[:, 1] = pred
    df0 = pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close'])

    new_pred = [pred[0]]
    for i in range(1, len(pred)):

        df = df0[:i].reset_index(drop=True)
        p = calc_profit(df)

        if p < th:
            if pred[i] == 1:
                new_pred.append(0)
            elif pred[i] == 0:
                new_pred.append(1)
            else:
                new_pred.append(1)
        else:
            new_pred.append(pred[i])

    return np.array(new_pred)
