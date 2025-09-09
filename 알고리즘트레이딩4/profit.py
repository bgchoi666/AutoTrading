# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# non overnight, 수수료 0.0015% (살때, 팔때 각각)
# 매도 : pred = 0, state = 1
# 매수 : pred = 1, state = 2
# 중립 : state = 0

import pandas as pd
import numpy as np

import datetime
import sys

result_path = 'ensemble/pred_83_results.csv'

pred_term = 5

loss_cut = 0.001
profit_cut = 0.01

trading_9h = False

def calc_profit():
    df = pd.read_csv(result_path, encoding='euc-kr')

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
                state = pred + 1
                count = 1
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
                state = pred + 1
                count = 1
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

    df['profit'] = profit
    df['fee'] = fee

    df.to_csv(result_path, index=False, encoding='euc-kr')

    return (sum(profit) - sum(fee))/(df['close'].values.mean()*250000*0.078) + 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pred_term = int(sys.argv[1])
    r = calc_profit()
    print(r)