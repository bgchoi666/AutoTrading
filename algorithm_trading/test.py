# -*- coding:utf-8 -*-

import pandas as pd
import logging
import shutil
import datetime
import save_trans as st
import numpy as np
import random
import os

os.system("ftp_download_from_PC2.bat")
exit(0)

st.save_trans("매도", 1, 346)

exit(0)

now = datetime.datetime.now()
trading_date = now.strftime("%Y/%m/%d/%H:%M")

if trading_date[11:13] < '15':
    print("loss_cut으로 인한 프로그램 종료")
    logging.info("loss_cut으로 인한 프로그램 종료")
    exit(0)

exit(0)

trans_df = pd.read_csv("trans_20.csv", encoding="euc-kr")

now = datetime.datetime.now()
trans_df = trans_df.loc[trans_df["거래일시"] >= now.strftime("%Y/%m/%d/09:00")]

num = trans_df.loc[trans_df["매매구분"] == "매수", "수량"].values.sum() - \
      trans_df.loc[trans_df["매매구분"] == "매도", "수량"].values.sum()

exit(0)

M_df = pd.read_csv("../알고리즘트레이딩4/kospi200f_11_60M.csv", encoding='euc-kr')
last_day_df = M_df.loc[M_df['date'] >= M_df['date'].values[-1][:10]]

date = datetime.datetime.strptime(last_day_df['date'].values[0], "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d")
open = last_day_df['시가'].values[0]
high = last_day_df['고가'].values.max()
low = last_day_df['저가'].values.min()
close = last_day_df['종가'].values[-1]
volume = last_day_df['거래량'].values.sum()
outs = last_day_df['미결제'].values[-1]

columns = ['date', '시가', '고가', '저가', '종가', 'PDI', 'MDI', 'ADX', 'SlowK', 'SlowD', '거래량', '미결제']
new_df = pd.DataFrame(columns=columns)

new_data = {'date': date, '시가': open, '고가': high, '저가': low,
            '종가': close, 'PDI': 0, 'MDI': 0, 'ADX': 0, 'SlowK': 0, 'SlowD': 0,
            '거래량': volume, '미결제': outs}

new_df = new_df.append(new_data, ignore_index=True)

df = pd.read_csv("../알고리즘1D/kospi200f_1D.csv", encoding='euc-kr')

for j in range(len(new_df)):
    # 하나씩 df에 붙여서 계산하기
    a = np.reshape(new_df.values[j, :], (1, 12))
    a_df = pd.DataFrame(np.array(a), columns=columns)
    df = pd.concat([df, a_df], axis=0, ignore_index=True)

    # DMI, ADX 값 계산
    TR = np.zeros(14)
    DM_plus = np.zeros(14)
    DM_minus = np.zeros(14)
    for i in range(14):
        DM_plus[i] = max(max(df['고가'].values[-(i + 1)] - df['고가'].values[-(i + 2)], 0) -
                         max(df['저가'].values[-(i + 2)] - df['저가'].values[-(i + 1)], 0), 0)
        DM_minus[i] = max(max(df['저가'].values[-(i + 2)] - df['저가'].values[-(i + 1)], 0) -
                          max(df['고가'].values[-(i + 1)] - df['고가'].values[-(i + 2)], 0), 0)
        TR[i] = max(abs(df['고가'].values[-(i + 1)] - df['저가'].values[-(i + 1)]),
                    abs(df['고가'].values[-(i + 1)] - df['종가'].values[-(i + 2)]),
                    abs(df['저가'].values[-(i + 1)] - df['종가'].values[-(i + 2)]), )
    TR_avg = np.mean(TR)
    DM_plus_avg = np.mean(DM_plus)
    DM_minus_avg = np.mean(DM_minus)

    PDI = DM_plus_avg / TR_avg * 100
    MDI = DM_minus_avg / TR_avg * 100
    ADX = abs(PDI - MDI) / (PDI + MDI) * 100

    df['PDI'].values[-1] = PDI
    df['MDI'].values[-1] = MDI
    df['ADX'].values[-1] = ADX

    # SlowK, SlowD 계산
    FastK = np.zeros(10)
    FastD = np.zeros(5)
    for i in range(10):
        FastK[i] = (df['종가'].values[len(df) - i - 1] - min(df['저가'].values[-i - 12:len(df) - i])) / (
                max(df['고가'].values[-i - 12:(len(df) - i)]) - min(
            df['저가'].values[-i - 12:len(df) - i])) * 100
    for i in range(5):
        FastD[i] = np.mean(FastK[i:i + 5])
    df.loc[len(df) - 1, 'SlowK'] = np.mean(FastD[:5])
    df.loc[len(df) - 1, 'SlowD'] = np.mean(df['SlowK'].values[-5:])

df.to_csv("../알고리즘1D/kospi200f_1D_s.csv", index=False, encoding='euc-kr')

exit(0)

trans_df = pd.read_csv("trans_2.csv", encoding='euc-kr')

# 오늘 잔고 수량과 매수매도 상태
now = datetime.datetime.now()
trans_df_today = trans_df.loc[trans_df["거래일시"] >= "2023/02/21/09:00"]
trans_df_today = trans_df_today.loc[trans_df_today["거래일시"] <= "2023/02/21/15:50"]
print(trans_df_today)
num = trans_df_today.loc[trans_df_today["매매구분"] == "매수", "수량"].values.sum() - \
      trans_df_today.loc[trans_df_today["매매구분"] == "매도", "수량"].values.sum()
print(num)

exit(0)

# 오늘 잔고 수량과 매수매도 상태
now = datetime.datetime.now()
trans_df_today = trans_df.loc[trans_df["거래일시"] >= now.strftime("%Y/%m/%d/09:00")]

num = trans_df_today.loc[trans_df_today["매매구분"] == "매수", "수량"].values.sum() - \
      trans_df_today.loc[trans_df_today["매매구분"] == "매도", "수량"].values.sum()

a = np.array([1, 2, 3])
b = np.array([3, 2, 1])

print(np.std(a/b/2 + 1))

exit(0)

#result_path = "C:/Users/user/Desktop/알고리즘트레이딩4/pred_83_results.csv"
#results_df = pd.read_csv(result_path, encoding='euc-kr')
#v = results_df.loc[results_df.index.max()].values

v = [0, 0, 0, 0, 0]

r = np.unique(v[1:5], return_counts=True)
if len(r[1]) == 1:
    pred_prob = 1
else:
    pred_prob = 0.5

exit(0)

file_path = "kospi200f_11_60M.csv"
v4_path = "C:/Users/user/Desktop/알고리즘트레이딩4/kospi200f_11_60M.csv"
new_path = "C:/Users/user/Desktop/알고리즘트레이딩_new/kospi200f_11_60M.csv"

shutil.copy(file_path, v4_path)
shutil.copy(file_path, new_path)

exit(0)


start_time = datetime.now()

print((datetime.now() - start_time).seconds)

exit(0)

dollar_results_df = pd.read_csv("C:/users/user/desktop/dollar_1D/models/doller_1D_results.csv", encoding='euc-kr')
buy_sell_signal = int(dollar_results_df['results'].values[-1])
if buy_sell_signal == 1:
    gubun = "매수"
else:
    gubun = "매도"

# 주문 내역 거래내역 파일 (dollar_trans.csv) 에 저장
trans_df = pd.read_csv('dollar_trans.csv', encoding='euc-kr')

if int(trans_df.loc[trans_df.index.max(), "청산일시"]) != 0:
    state = 0
else:
    state = trans_df.loc[trans_df.index.max(), "매매구분"]

if state == 0:

    # 신규 진입
    trans_data = {'거래일시': '2022-07-01', '매매구분': gubun, '수량': 1,
                  '거래가격': '0000', '청산일시': '0', '청산가격': '0'}
    trans_df = trans_df.append(trans_data, ignore_index=True)
    trans_df.to_csv('dollar_trans.csv', index=False, encoding='euc-kr')

    print(trans_data)
    print('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')

elif gubun != state:

    # 청산
    if gubun == "매도":
        trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산일시'] = '2022-07-01'
        trans_df.loc[trans_df.oc[trans_df["매매구분"] == "매수"].index.max(), '청산가격'] = '0000'
    else:
        trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매도"].index.max(), '청산일시'] = '2022-07-01'
        trans_df.loc[trans_df.oc[trans_df["매매구분"] == "매도"].index.max(), '청산가격'] = '0000'

    # 신규 진입
    trans_data = {'거래일시': '2022-07-01', '매매구분': gubun, '수량': 1,
                  '거래가격': '0000', '청산일시': '0', '청산가격': '0'}
    trans_df = trans_df.append(trans_data, ignore_index=True)
    trans_df.to_csv('dollar_trans.csv', index=False, encoding='euc-kr')

    print(trans_data)
    print('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
else:
    print('달러 같은 매매구분 ... 거래 없음')


"""
import pandas as pd
import numpy as np
import datetime
import openpyxl

a = np.array([1, 2, 1])
print(a.argmin())

now = datetime.datetime.now()
trans_df = pd.read_csv('trans_old.csv', encoding='euc-kr')

trans_df.loc[trans_df.index.max(), '청산가격'] = trans_df.loc[trans_df.index.max(), '거래가격']
trans_df.loc[trans_df.index.max(), '청산일시'] = now.strftime("%Y/%m/%d/%H:%M")
trans_data = {'거래일시': 'test', '매매구분': '매수', '수량': 1,
              '거래가격': 1000000, '청산일시': '0', '청산가격': '0'}
trans_df = trans_df.append(trans_data, ignore_index=True)
trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')

#!/usr/bin/env python3
import sys, datetime
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtCore import pyqtSlot, QTimer


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 윈도우 설정
        self.setGeometry(300, 300, 400, 300)  # x, y, w, h
        self.setWindowTitle('Timer Window')

        # 시간을 표시할 라벨 생성
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 200, 20)

        # Timer1 설정
        self.timer_one = QTimer(self)
        self.timer_one.start(3000)
        self.timer_one.timeout.connect(self.timeout_run)

        # Timer2 설정
        self.timer_two = QTimer(self)
        self.timer_two.start(3000)
        self.timer_two.timeout.connect(self.timeout_run)

    @pyqtSlot()
    def timeout_run(self):
        # 2개의 타이머를 구분하기 위한 객체
        sender = self.sender();
        current_time = datetime.datetime.now()

        if id(sender) == id(self.timer_one):
            print("timer1 call --> ", current_time)
        elif id(sender) == id(self.timer_two):
            print("timer2 call --> ", current_time)

        self.label.setText(str(current_time))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

data = '-1,234'
strip_data = data.lstrip('-0')
if strip_data == '':
    strip_data = '0'

try:
    format_data = format(int(strip_data), ',d')
except:
    format_data = format(float(strip_data))

if data.startswith('-'):
    format_data = '-' + format_data

print(format_data)

import numpy as np
data = np.load('c:/users/user/desktop/정재용/KRW-BTC.npy')
print(data)


import datetime

now = datetime.datetime.now()
trans_date = '2021/07/30/11:00'
trans_date = datetime.datetime.strptime(trans_date, '%Y/%m/%d/%H:%M')

print(now.day - trans_date.day)
print( (15 - trans_date.hour + 1 + now.hour - 9) * 3600 + now.minute * 60)
print(now.month)

if ((now.day == trans_date.day and (now - trans_date).seconds >= 18000) or
        ((now.day - trans_date.day > 0 or now.month > trans_date.month or now.year > trans_date.year) and
         ((15 - trans_date.hour + 1 + now.hour - 9) * 3600 + now.minute * 60 >= 18000))):
    print('ok')
"""

