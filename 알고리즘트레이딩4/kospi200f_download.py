# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 키움 API를 이용한 코스피 200 선물 거래 시스템
# 앙상블 모델을 이용
# version4 : DMI, stochastic 항목 5개 추가 (non-normalization)

import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import datetime
from datetime import timedelta
import time
import os
import pandas as pd
import numpy as np
import logging
import pandas_datareader.data as web
from datetime import date
from openpyxl import load_workbook

form_class = uic.loadUiType("futureTrader.ui")[0]
file_path = "kospi200f_11_60M.csv"
result_path = "pred_83_results.csv"

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()

        self.setupUi(self)

        self.kiwoom = Kiwoom()
        # login 화면 연결
        self.kiwoom.comm_connect()

        accouns_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
        accounts = self.kiwoom.get_login_info("ACCNO")

        accounts_list = accounts.split(';')[0:accouns_num]

        self.acc_no = accounts_list[0]
        if self.acc_no[8:] != '31':
            self.acc_no = accounts_list[1]

        self.comboBox.addItems(accounts_list)

        self.loss_cut = self.lineEdit.text()
        self.profit_real = self.lineEdit_3.text()

        futureCodes = self.kiwoom.dynamicCall("GetFutureList()")
        self.futureCode = futureCodes.split(";")[0]

        self.update_data()

    def update_data(self):
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

        df = pd.read_csv(file_path, encoding='euc-kr')
        last_date = df.loc[df.index.max(), 'date']

        if QTime.currentTime() <= QTime(15, 45, 0) and self.kiwoom.che_time <= last_date:
            return
        elif QTime.currentTime() > QTime(15, 45, 0) and last_date[11:13] == '15' and self.kiwoom.che_time <= last_date:
            return

        new_data = []
        # 장 종료후 마지막 시간대 추가
        if QTime.currentTime() > QTime(15, 45, 0):
            n = 0
        else:
            n = 1

        columns = ['date', '시가', '고가', '저가', '종가', 'PDI', 'MDI', 'ADX', 'SlowK', 'SlowD', '거래량', '미결제']
        new_df = pd.DataFrame(columns=columns)
        while self.kiwoom.opt50029_output['multi'][n][0] > last_date:

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

        df.to_csv(file_path, index=False, encoding='euc-kr')

        print(datetime.datetime.now())
        print("new data appended....")
        print(new_data)

        exit(0)

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

        if rqname == "opt50001_req": # 선옵현재가 정보
            self._opt50001(rqname, trcode)
        elif rqname == "가격거래량_req":
            self._OPT50029(rqname, trcode)
        elif rqname == "미결제약정_req":
            self._OPT50062(rqname, trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass


    def _OPT50029(self, rqname, trcode):
        self.reset_opt50029_output()

        # 시가, 고가, 저가, 종가, 거래량
        self.close = (self._comm_get_data(trcode, "", rqname, 1, "현재가")).lstrip('+-')
        self.open = (self._comm_get_data(trcode, "", rqname, 1, "시가")).lstrip('+-')
        self.high = (self._comm_get_data(trcode, "", rqname, 1, "고가")).lstrip('+-')
        self.low = (self._comm_get_data(trcode, "", rqname, 1, "저가")).lstrip('+-')
        self.volumn = self._comm_get_data(trcode, "", rqname, 1, "거래량").replace(',', '')

        t = str(self._comm_get_data(trcode, "", rqname, 1, "체결시간"))
        self.che_time = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

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

            self.opt50029_output['multi'].append([t, open, high, low, close, volumn])

    def _OPT50062(self, rqname, trcode):
        self.reset_opt50062_output()

        self.open_interest = self._comm_get_data(trcode, "", rqname, 0, "미결제약정")

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        cnt = min(rows, 50)
        for i in range(cnt):
            # 미결제약정
            open_interest = self._comm_get_data(trcode, "", rqname, i, "미결제약정").lstrip('+-')

            t = str(self._comm_get_data(trcode, "", rqname, i, "체결시간"))
            t = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

            self.opt50062_output['multi'].append([t, open_interest])

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

    def reset_sell_output(self):
        self.sell_output = {'single': [], 'multi': []}

    def get_server_gubun(self):
        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()