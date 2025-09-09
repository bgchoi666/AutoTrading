# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 코스피 선물, 카카오, 삼성 데이터 최근일까지 갱신


import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import datetime
from datetime import timedelta
import time
import os
import shutil
import pandas as pd
import numpy as np
import logging

import pandas_datareader.data as web
from datetime import date
from openpyxl import load_workbook

#form_class = uic.loadUiType("futureTrader.ui")[0]
file_path = "H:/알고리즘트레이딩2/kospi200f_11_60M.csv"
v4_path = "H:/알고리즘트레이딩4/kospi200f_11_60M.csv"#version 4
new_path = "H:/알고리즘트레이딩_new/kospi200f_11_60M.csv"#version 'new'
알고리즘1D_path = "H:/알고리즘1D/kospi200f_1D.csv"
#log_path = "futureTrader60M.log"


class MyWindow(QMainWindow):#, form_class):
    def __init__(self, gubun):
        super().__init__()

        #self.setupUi(self)

        self.gubun = gubun

        #logging.basicConfig(filename='futureTrader60M.log', level=logging.DEBUG)

        self.kiwoom = Kiwoom()
        # login 화면 연결
        self.kiwoom.comm_connect()

        accouns_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
        accounts = self.kiwoom.get_login_info("ACCNO")

        accounts_list = accounts.split(';')[0:accouns_num]
        self.acc_no = accounts_list[2]

        #self.comboBox.addItems(accounts_list)

        #self.loss_cut = self.lineEdit.text()
        #self.profit_real = self.lineEdit_3.text()

        futureCodes = self.kiwoom.dynamicCall("GetFutureList()")
        self.futureCode = futureCodes.split(";")[0]

        # 삼성전자, 카카오 데이터를 받기 위하여
        self.samsungCode = '005930'
        self.kakaoCode = '035720'
        self.dollarCode = '175TA000'

        self.samsung_path = "F:/키움_주식_달러선물/samsung_60M.csv"
        self.kakao_path = "F:/키움_주식_달러선물/kakao_60M.csv"
        self.dollar_path = "H:/dollar_1D/dollar_1D.csv"
        self.kakao_1D_path = "H:/stock_1D/kakao_1D.csv"
        self.samsung_1D_path = "H:/stock_1D/samsung_1D.csv"

        if gubun == '1':
            # Timer1
            self.timer = QTimer(self)
            self.timer.start(1000*60)
            self.timer.timeout.connect(self.timeout)
        else:
            self.timeout()
    def timeout(self):

        #거래 가능 시간 check
        if self.gubun == '1':
            if QTime.currentTime() > QTime(16, 0, 0) or QTime.currentTime() < QTime(8, 0, 0):
                print("거래가능시간아님")
                sys.exit(0)

        # 매시간 마다 코스피선물, 카카오, 삼성전자 데이터 다운로드, csv 파일에 저장
        now = datetime.datetime.now()
        m = now.minute
        h = now.hour
        if self.gubun == '1':
            if h < 9 or h > 15 or m != 0:
                return

        self.update_data()
        self.update_v4_data()
        shutil.copy(v4_path, new_path)
        #self.stock_update_data(self.samsungCode, self.samsung_path)
        #self.stock_update_data(self.kakaoCode, self.kakao_path)

        if h > 15 or (h == 15 and m > 45):
            self.dollar_update(self.dollarCode, self.dollar_path)
            self.stock_1D_update(self.samsungCode, self.samsung_1D_path)
            self.stock_1D_update(self.kakaoCode, self.kakao_1D_path)
            self.알고리즘1D_update()

        if self.gubun == '0':
            exit(0)

    def update_data(self):
        # 선물 시가, 고가, 저가, 현재가(종가), 거래량
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("시간단위", "60")
        self.kiwoom.comm_rq_data("가격거래량_req", "opt50029", "0", "3001")
        time.sleep(0.2)

        print('가격, 거래량 조회 완료...')

        # 선물 미결제 약정
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("시간단위", "60")
        self.kiwoom.comm_rq_data("미결제약정_req", "opt50062", "0", "3002")
        time.sleep(0.2)

        print('미결제 약정 조회 완료...')

        df = pd.read_csv(file_path, encoding='euc-kr')
        last_date = df.loc[df.index.max(), 'date']

        if self.kiwoom.che_time <= last_date:
            print("알고리즘트레이딩2 추가 데이터 없음")
            return

        new_data = []
        n = 1
        while self.kiwoom.opt50029_output['multi'][n][0] > last_date and self.kiwoom.opt50029_output['multi'][n][0][11:13] != '16':
            data = self.kiwoom.opt50029_output['multi'][n][:5]
            m = df.index.max()
            data.append((df.loc[m-3:, '종가'].sum() + float(data[4])) / 5)
            data.append((df.loc[m-18:, '종가'].sum() + float(data[4])) / 20)
            data.append((df.loc[m-58:, '종가'].sum() + float(data[4])) / 60)
            data.append((df.loc[m-118:, '종가'].sum() + float(data[4])) / 120)
            data.append((df.loc[m-198:, '종가'].sum() + float(data[4])) / 200)
            data.append(self.kiwoom.opt50029_output['multi'][n][5])
            data.append(self.kiwoom.opt50062_output['multi'][n][1])
            new_data.append(data)
            n += 1

        new_data = sorted(new_data, key=lambda x: x[0])
        if new_data == []:
            return
        new_df = pd.DataFrame(np.array(new_data), columns=['date', '시가', '고가', '저가', '종가',
                                                           '5', '20', '60', '120', '200', '거래량', '미결제'])


        df = pd.concat([df, new_df], axis=0, ignore_index=True)

        df.to_csv(file_path, index=False, encoding='euc-kr')

        print(datetime.datetime.now())
        logging.info(datetime.datetime.now())
        print("new data appended....")
        print(new_data)
        logging.info(new_data)

    def update_v4_data(self):
        # 선물 시가, 고가, 저가, 현재가(종가), 거래량
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("시간단위", "60")
        self.kiwoom.comm_rq_data("가격거래량_req", "opt50029", "0", "3001")
        time.sleep(0.2)

        print('가격, 거래량 조회 완료...')

        # 선물 미결제 약정
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("시간단위", "1800")
        self.kiwoom.comm_rq_data("미결제약정_req", "opt50062", "0", "3002")
        time.sleep(0.2)

        print('미결제 약정 조회 완료...')

        file_path4 = "H:/알고리즘트레이딩4/kospi200f_11_60M.csv"
        df = pd.read_csv(file_path4, encoding='euc-kr')
        last_date = df.loc[df.index.max(), 'date']

        if QTime.currentTime() <= QTime(15, 45, 0) and self.kiwoom.che_time <= last_date:
            return
        elif QTime.currentTime() > QTime(15, 45, 0) and last_date[11:13] == '15' and self.kiwoom.che_time <= last_date:
            print("알고리즘트레이딩4 추가 데이터 없음")
            return

        #new_data = []
        # 장 종료후 마지막 시간대 추가
        #if QTime.currentTime() > QTime(15, 45, 0) or QTime.currentTime() < QTime(9, 0, 0):
        #    n = 0
        #else:
        #    n = 1

        columns = ['date', '시가', '고가', '저가', '종가', 'PDI', 'MDI', 'ADX', 'SlowK', 'SlowD', '거래량', '미결제']
        new_df = pd.DataFrame(columns=columns)

        n = 1
        while self.kiwoom.opt50029_output['multi'][n][0] > last_date and self.kiwoom.opt50029_output['multi'][n][0][11:13] != '16':

            # 데이터 추가
            new_data = {'date': self.kiwoom.opt50029_output['multi'][n][0], '시가': float(self.kiwoom.opt50029_output['multi'][n][1]),
                        '고가': float(self.kiwoom.opt50029_output['multi'][n][2]), '저가': float(self.kiwoom.opt50029_output['multi'][n][3]),
                        '종가': float(self.kiwoom.opt50029_output['multi'][n][4]), 'PDI': 0, 'MDI': 0, 'ADX': 0, 'SlowK': 0, 'SlowD': 0,
                        '거래량': float(self.kiwoom.opt50029_output['multi'][n][5]), '미결제': float(self.kiwoom.opt50062_output['multi'][n][1])}

            new_df = new_df.append(new_data, ignore_index=True)

            n += 1

        new_df = new_df.sort_values("date", axis=0).reset_index(drop=True)

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
                DM_plus[i] = max(max(df['고가'].values[-(i+1)] - df['고가'].values[-(i + 2)], 0) -
                                 max(df['저가'].values[-(i + 2)] - df['저가'].values[-(i+1)], 0), 0)
                DM_minus[i] = max(max(df['저가'].values[-(i + 2)] - df['저가'].values[-(i+1)], 0) -
                                  max(df['고가'].values[-(i+1)] - df['고가'].values[-(i + 2)], 0), 0)
                TR[i] = max(abs(df['고가'].values[-(i+1)] - df['저가'].values[-(i+1)]),
                            abs(df['고가'].values[-(i+1)] - df['종가'].values[-(i + 2)]),
                            abs(df['저가'].values[-(i+1)] - df['종가'].values[-(i + 2)]), )
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

        df.to_csv(file_path4, index=False, encoding='euc-kr')

        # save 마지막 종가
        self.current_price = float(df['종가'].values[-1])

    def 알고리즘1D_update(self):
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        df = pd.read_csv(알고리즘1D_path, encoding='euc-kr')

        if df['date'].values[-1] >= today:
            print("알고리즘_1D 추가 데이터 없음")
            return

        M_df = pd.read_csv(v4_path, encoding='euc-kr')
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

        df.to_csv(알고리즘1D_path, index=False, encoding='euc-kr')

    def stock_update_data(self, code, path):
        # 주식 시가, 고가, 저가, 현재가(종가), 거래량s
        self.kiwoom.reset_opt10080_output()
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("틱범위", "60")
        self.kiwoom.comm_rq_data("주식데이터_req", "opt10080", "0", "4001")
        time.sleep(0.2)

        if not self.kiwoom.opt10080_output['multi']:
            print(code + " update 실패")
            return

        df = pd.read_csv(path, encoding='euc-kr')
        last_date = df.loc[df.index.max(), 'date']
        current_time = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%Y/%m/%d/%H:%M")
        if QTime.currentTime() <= QTime(15, 45, 0) and current_time <= last_date:
            return
        elif QTime.currentTime() > QTime(15, 45, 0) and last_date[11:13] == '15' and current_time <= last_date:
            return

        new_data = []
        # 장 종료후 마지막 시간대 추가
        if QTime.currentTime() > QTime(15, 45, 0):
            n = 0
        else:
            n = 1
        while self.kiwoom.opt10080_output['multi'][n][0] > last_date:
            new_data.append(self.kiwoom.opt10080_output['multi'][n])
            n += 1
        new_data = sorted(new_data, key=lambda x: x[0])
        if new_data == []:
            print(code + ":60M 추가 데이터 없음")
            return
        new_df = pd.DataFrame(np.array(new_data), columns=['date', '시가', '고가', '저가', '종가', '거래량'])
        new_df = pd.concat([df, new_df], axis=0, ignore_index=True)

        new_df.to_csv(path, index=False, encoding='euc-kr')

        print(code + ': 데이터 추가 완료...')

    def dollar_update(self, code, path):

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # ========== 데이터 파일 update ===============================
        df = pd.read_csv(path, encoding='euc-kr')

        if df['date'].values[-1] >= today:
            print("dollar_1D 추가 데이터 없음")
            return

        # 가격 데이터 가져오기
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("OPT50019_req", "OPT50019", 0, "5000")
        time.sleep(0.2)

        # 데이터 추가
        new_data = {'date': datetime.datetime.now().strftime('%Y-%m-%d'), '시가': self.kiwoom.dollar_open,
                    '고가': self.kiwoom.dollar_high, '저가': self.kiwoom.dollar_low, '종가': self.kiwoom.dollar_close,
                    'PDI': 0, 'MDI': 0, 'ADX': 0}

        df = df.append(new_data, ignore_index=True)

        # DMI, ADX 값 계산
        TR = np.zeros(14)
        DM_plus = np.zeros(14)
        DM_minus = np.zeros(14)
        for i in range(14):
            DM_plus[i] = max(max(df['고가'].values[-(i+1)] - df['고가'].values[-(i+2)], 0) -
                             max(df['저가'].values[-(i+2)] - df['저가'].values[-(i+1)], 0), 0)
            DM_minus[i] = max(max(df['저가'].values[-(i+2)] - df['저가'].values[-(i+1)], 0) -
                              max(df['고가'].values[-(i+1)] - df['고가'].values[-(i+2)], 0), 0)
            TR[i] = max(abs(df['고가'].values[-(i+1)] - df['저가'].values[-(i+1)]),
                        abs(df['고가'].values[-(i+1)] - df['종가'].values[-(i+2)]),
                        abs(df['저가'].values[-(i+1)] - df['종가'].values[-(i+2)]),)
        TR_avg = np.mean(TR)
        DM_plus_avg = np.mean(DM_plus)
        DM_minus_avg = np.mean(DM_minus)

        PDI = DM_plus_avg / TR_avg * 100
        MDI = DM_minus_avg / TR_avg * 100
        ADX = abs(PDI - MDI) / (PDI + MDI) * 100

        df['PDI'].values[-1] = PDI
        df['MDI'].values[-1] = MDI
        df['ADX'].values[-1] = ADX


        df.to_csv(path, index=False, encoding='euc-kr')

    def stock_1D_update(self, code, path):

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # ========== 데이터 파일 update ===============================
        df = pd.read_csv(path, encoding='euc-kr')

        if df['date'].values[-1] >= today:
            print(code + ":_1D 추가 데이터 없음")
            return

        # 가격 데이터 가져오기
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "5001")
        time.sleep(0.2)

        # 데이터 추가
        new_data = {'date': datetime.datetime.now().strftime('%Y-%m-%d'), '시가': self.kiwoom.stock_open,
                    '고가': self.kiwoom.stock_high, '저가': self.kiwoom.stock_low, '종가': self.kiwoom.stock_close,
                    'PDI': 0, 'MDI': 0, 'ADX': 0, 'SlowK': 0, 'SlowD': 0, '거래량': self.kiwoom.stock_volume}

        df = df.append(new_data, ignore_index=True)

        # DMI, ADX 값 계산
        TR = np.zeros(14)
        DM_plus = np.zeros(14)
        DM_minus = np.zeros(14)
        for i in range(14):
            DM_plus[i] = max(max(df['고가'].values[-(i+1)] - df['고가'].values[len(df)-(i+2)], 0) -
                             max(df['저가'].values[-(i+2)] - df['저가'].values[-(i+1)], 0), 0)
            DM_minus[i] = max(max(df['저가'].values[-(i+2)] - df['저가'].values[-(i+1)], 0) -
                              max(df['고가'].values[-(i+1)] - df['고가'].values[-(i+2)], 0), 0)
            TR[i] = max(abs(df['고가'].values[-(i+1)] - df['저가'].values[-(i+1)]),
                        abs(df['고가'].values[-(i+1)] - df['종가'].values[-(i+2)]),
                        abs(df['저가'].values[-(i+1)] - df['종가'].values[-(i+2)]),)
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
            FastK[i] = (df['종가'].values[-1] - min(df['저가'].values[-i-12:len(df)-i]))/(max(df['고가'].values[-1-12:(len(df)-i)]) - min(df['저가'].values[-i-12:len(df)-i])) * 100
        for i in range(5):
            FastD[i] = np.mean(FastK[i:i+5])
        df.loc[len(df)-1, 'SlowK'] = np.mean(FastD[:5])
        df.loc[len(df)-1, 'SlowD'] = np.mean(df['SlowK'].values[-5:])

        df.to_csv(path, index=False, encoding='euc-kr')


TR_RE_TIME_INTERVAL = 0.2

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()

        self.calcul_data = []

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)
        self.OnReceiveChejanData.connect(self._receive_chejan_data)

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _event_connect(self, err_code):
        if err_code == 0:
            print("connected")
        else:
            print("disconnected")

        self.login_event_loop.exit()

    def get_future_list(self):
        future_list = self.dynamicCall("GetFutureList()")
        future_list = future_list.split(';')
        return future_list

    def get_option_code(self, price, gubun, magam):
        code = self.dynamicCall("GetOptionCodeName(QString, int, QString)", price, gubun, magam)
        return code

    def get_connect_state(self):
        ret = self.dynamicCall("GetConnectState()")
        return ret

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString)", rqname, trcode, next, screen_no)
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()

    def _comm_get_data(self, code, real_type, field_name, index, item_name):
        ret = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", code,
                               real_type, field_name, index, item_name)
        return ret.strip()

    def _get_repeat_cnt(self, trcode, rqname):
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False

        if rqname == "가격거래량_req":
            self._OPT50029(rqname, trcode)
        elif rqname == "미결제약정_req":
            self._OPT50062(rqname, trcode)
        elif rqname == "주식데이터_req":
            self._OPT10080(rqname, trcode)
        elif rqname == "OPT50019_req":
            self._OPT50019(rqname, trcode)
        elif rqname == "opt10001_req":
            self._opt10001(rqname, trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass


    def _OPT10080(self, rqname, trcode):
        self.reset_opt10080_output()

        # single data
        code = self._comm_get_data(trcode, "", rqname, 0, "종목코드")
        self.opt10080_output['single'].append(code)

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        cnt = min(rows, 50)
        for i in range(cnt):
            # 시가, 고가, 저가, 종가, 거래량
            close = (self._comm_get_data(trcode, "", rqname, i, "현재가")).lstrip('+-')
            open = (self._comm_get_data(trcode, "", rqname, i, "시가")).lstrip('+-')
            high = (self._comm_get_data(trcode, "", rqname, i, "고가")).lstrip('+-')
            low = (self._comm_get_data(trcode, "", rqname, i, "저가")).lstrip('+-')
            volumn = self._comm_get_data(trcode, "", rqname, i, "거래량").replace(',', '')

            t = str(self._comm_get_data(trcode, "", rqname, i, "체결시간"))
            t = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

            self.opt10080_output['multi'].append([t, open, high, low, close, volumn])


    def _OPT50029(self, rqname, trcode):
        self.reset_opt50029_output()

        # 시가, 고가, 저가, 종가, 거래량
        self.close = (self._comm_get_data(trcode, "", rqname, 1, "현재가")).lstrip('+-')
        self.open = (self._comm_get_data(trcode, "", rqname, 1, "시가")).lstrip('+-')
        self.high = (self._comm_get_data(trcode, "", rqname, 1, "고가")).lstrip('+-')
        self.low = (self._comm_get_data(trcode, "", rqname, 1, "저가")).lstrip('+-')
        self.volumn = self._comm_get_data(trcode, "", rqname, 1, "거래량").replace(',', '')

        self.che_time = (datetime.datetime.strptime(self._comm_get_data(trcode, "", rqname, 1, "체결시간"), "%Y%m%d%H%M%S") + datetime.timedelta(hours=1)).strftime("%Y/%m/%d/%H:00")
        #self.che_time = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        cnt = min(rows, 50)
        for i in range(cnt):
            # 시가, 고가, 저가, 종가, 거래량
            close = (self._comm_get_data(trcode, "", rqname, i, "현재가")).lstrip('+-')
            open = (self._comm_get_data(trcode, "", rqname, i, "시가")).lstrip('+-')
            high = (self._comm_get_data(trcode, "", rqname, i, "고가")).lstrip('+-')
            low = (self._comm_get_data(trcode, "", rqname, i, "저가")).lstrip('+-')
            volumn = self._comm_get_data(trcode, "", rqname, i, "거래량").replace(',', '')

            t = (datetime.datetime.strptime(self._comm_get_data(trcode, "", rqname, i, "체결시간"),
                                        "%Y%m%d%H%M%S") + datetime.timedelta(hours=1)).strftime("%Y/%m/%d/%H:00")

            self.opt50029_output['multi'].append([t, open, high, low, close, volumn])

    def _OPT50062(self, rqname, trcode):
        self.reset_opt50062_output()

        self.open_interest = self._comm_get_data(trcode, "", rqname, 0, "미결제약정")

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        cnt = min(rows, 50)
        for i in range(cnt):
            # 미결제약정
            open_interest = (self._comm_get_data(trcode, "", rqname, i, "미결제약정")).lstrip('+-')

            t = str(self._comm_get_data(trcode, "", rqname, i, "체결시간"))
            t = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

            self.opt50062_output['multi'].append([t, open_interest])

    def _OPT50019(self, rqname, trcode):

        # 시가, 고가, 저가, 종가
        self.dollar_open = float((self._comm_get_data(trcode, "", rqname, 0, "시가")).lstrip('+-'))
        self.dollar_high = float((self._comm_get_data(trcode, "", rqname, 0, "고가")).lstrip('+-'))
        self.dollar_low = float((self._comm_get_data(trcode, "", rqname, 0, "저가")).lstrip('+-'))
        self.dollar_close = float((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))

    def _opt10001(self, rqname, trcode):

        self.item_name = self._comm_get_data(trcode, "", rqname, 0, "종목명")

        # 시가, 고가, 저가, 종가
        self.stock_open = int((self._comm_get_data(trcode, "", rqname, 0, "시가")).lstrip('+-'))
        self.stock_high = int((self._comm_get_data(trcode, "", rqname, 0, "고가")).lstrip('+-'))
        self.stock_low = int((self._comm_get_data(trcode, "", rqname, 0, "저가")).lstrip('+-'))
        self.stock_close = int((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))
        self.stock_volume = int((self._comm_get_data(trcode, "", rqname, 0, "거래량")).lstrip('+-'))

    def get_chejan_data(self, fid):
        ret = self.dynamicCall("GetChejanData(int)", fid)
        return ret

    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        print(gubun)

        print(self.get_chejan_data(302))
        print(self.get_chejan_data(10))
        print(self.get_chejan_data(930))
        print(self.get_chejan_data(931))

    def get_login_info(self, tag):
        ret = self.dynamicCall("GetLoginInfo(QString)", tag)
        return ret

    @staticmethod
    def change_format(data):
        strip_data = data.lstrip('-+0')
        if strip_data == '':
            strip_data = '0'

        try:
            format_data = format(int(strip_data), ',d')
        except:
            format_data = format(float(strip_data))

        if data.startswith('-'):
            format_data = '-' + format_data

        return format_data

    @staticmethod
    def change_format2(data):
        strip_data = data.lstrip('-0')

        if strip_data == '':
            strip_data = '0'

        if strip_data.startswith('.'):
            strip_data = '0' + strip_data

        if data.startswith('-'):
            strip_data = '-' + strip_data

        return strip_data

    def reset_opt50062_output(self):
        self.opt50062_output = {'multi': []}

    def reset_opt50029_output(self):
        self.opt50029_output = {'multi': []}

    def reset_opt10080_output(self):
        self.opt10080_output = {'single': [], 'multi': []}

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow(sys.argv[1])
    myWindow.show()
    app.exec_()