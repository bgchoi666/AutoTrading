# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 키움 API를 이용한 코스피 200 선물 거래 시스템
# 앙상블 모델을 이용

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

form_class = uic.loadUiType("futureTrader.ui")[0]
dollar_60M_trans_path = "dollar_60M_trans.csv"
dollar_60M_result_path = "F:/dollar_60M/pred_50_results.csv"

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
        self.lineEdit_2.setText(str(self.futureCode))


        # 삼성전자, 카카오 데이터를 받기 위하여
        self.dollarCode = '175SA000'

        self.dollar_60M_path = "F:/dollar_60M/dollar_60M.csv"

        self.save_open = 0
        self.save_high = 0
        self.save_low = 0
        self.save_close = 0

        # Timer2
        self.timer2 = QTimer(self)
        self.timer2.start(60)
        self.timer2.timeout.connect(self.timeout2)

    def timeout2(self):

        # 달러 데이터 다운로드
        self.dollar_update_trade(self.dollarCode, self.dollar_60M_path)

    def dollar_update_trade(self, code, path):

        # 현재 60분봉 데이터(가격, 거래량, 미결제약정) 불러와서 매시간 1분 경과후 csv 파일에 저장, 예측 결과에 따라 주문
        now = datetime.datetime.now()
        m = now.minute
        h = now.hour

        #if h > 15 or h < 9:
        #    print("거래 시간이 아님")
        #    return

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # ========== 데이터 파일 update ===============================
        df = pd.read_csv(path, encoding='euc-kr')

        # 가격 데이터 가져오기
        self.kiwoom.reset_dollar()
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("OPT50019_req", "OPT50019", 0, "5000")
        time.sleep(0.2)
        cnt = 0
        while not self.kiwoom.dollar_close:
            if cnt > 10:
                print("달러 가격 정보 다운로드 실패")
                logging.info("딜러 가격 정보 다운로드 실패")
                return
            self.kiwoom.set_input_value("종목코드", code)
            self.kiwoom.comm_rq_data("OPT50019_req", "OPT50019", 0, "5000")
            time.sleep(0.2)
            cnt += 1

        if self.save_open == 0:
            self.save_open = self.kiwoom.dollar_close
            self.save_high = self.kiwoom.dollar_close
            self.save_low = self.kiwoom.dollar_close
            self.save_close = self.kiwoom.dollar_close
            return
        elif m != 0:
            self.save_high = max(self.kiwoom.dollar_close, self.save_high)
            self.save_low = min(self.kiwoom.dollar_close, self.save_low)
            self.save_close = self.kiwoom.dollar_close
            return

        # 데이터 추가
        new_data = {'date': today, '시가': self.save_open,
                    '고가': self.save_high, '저가': self.save_low, '종가': self.save_close,
                    'PDI': 0, 'MDI': 0, 'ADX': 0}

        df = df.append(new_data, ignore_index=True)

        self.save_open = self.kiwoom.dollar_close
        self.save_high = self.kiwoom.dollar_close
        self.save_low = self.kiwoom.dollar_close
        self.save_close = self.kiwoom.dollar_close

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

        df.to_csv(path, index=False, encoding='euc-kr')

        """
        # ==================== 앙상블 예측 ================================
        os.system("start cmd /c \"dollar_60M_ensemble_predict.bat\"")
        time.sleep(0.5)

        now = datetime.datetime.now()

        while True:
            try:
                dollar_results_df = pd.read_csv(dollar_60M_result_path, encoding='euc-kr')
            except:
                print('pred_results.csv file read error...')
                if (datetime.datetime.now() - now).seconds > 60 * 10:
                    print('달러 예측 프로세스 실패... 시간 경과 10분... 프로그램 종료')
                    logging.error('달러 예측 프로세스 실패... 시간 경과 10분... 프로그램 종료')
                    return
                continue
            if dollar_results_df.loc[dollar_results_df.index.max(), 'dates'] == today:
                break
            if (datetime.datetime.now() - now).seconds > 60 * 10:
                print('달러 예측 프로세스 실패... 시간 경과 10분... 프로그램 종료')
                logging.error('달러 예측 프로세스 실패... 시간 경과 10분... 프로그램 종료')
                return

        # ==================== 거래 =======================================
        buy_sell_signal = int(dollar_results_df['results'].values[-1])
        if buy_sell_signal == 1:
            gubun = "매수"
            order_gubun = "2"
        else:
            gubun = "매도"
            order_gubun = "1"

        # 주문 내역 거래내역 파일 (dollar_trans.csv) 에 저장
        trans_df = pd.read_csv(dollar_60M_trans_path, encoding='euc-kr')

        if int(trans_df.loc[trans_df.index.max(), "청산일시"]) != 0:
            state = 0
        else:
            state = trans_df.loc[trans_df.index.max(), "매매구분"]

        if state == 0:

            # 신규 진입
            #self.kiwoom.send_order("send_order_req", "0101", self.acc_no, dollarCode, 1, order_gubun, "3", 1, "", "")
            trans_data = {'거래일시': today, '매매구분': gubun, '수량': 1,
                          '거래가격': self.kiwoom.dollar_close, '청산일시': '0', '청산가격': '0'}
            trans_df = trans_df.append(trans_data, ignore_index=True)
            trans_df.to_csv(dollar_60M_trans_path, index=False, encoding='euc-kr')

            print(trans_data)
            logging.info(trans_data)
            print('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')

        elif gubun != state:

            # 청산
            #self.kiwoom.send_order("send_order_req", "0101", self.acc_no, dollarCode, 1, order_gubun, "3", 1, "", "")
            if gubun == "매도":
                trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산일시'] = today
                trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산가격'] = self.kiwoom.dollar_close
            else:
                trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매도"].index.max(), '청산일시'] = today
                trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매도"].index.max(), '청산가격'] = self.kiwoom.dollar_close

            # 신규 진입
            #self.kiwoom.send_order("send_order_req", "0101", self.acc_no, dollarCode, 1, order_gubun, "3", 1, "", "")
            trans_data = {'거래일시': today, '매매구분': gubun, '수량': 1,
                          '거래가격': self.kiwoom.dollar_close, '청산일시': '0', '청산가격': '0'}
            trans_df = trans_df.append(trans_data, ignore_index=True)
            trans_df.to_csv(dollar_60M_trans_path, index=False, encoding='euc-kr')

            print(trans_data)
            logging.info(trans_data)
            print('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
        else:
            print('달러 같은 매매구분 ... 거래 없음')
            logging.info('달러 같은 매매구분 ... 거래 없음')
        """

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

        if rqname == "opt50001_req": # 선옵현재가 정보
            self._opt50001(rqname, trcode)
        elif rqname == "OPT50019_req":
            self._OPT50019(rqname, trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _opt50001(self, rqname, trcode):
        self.current_price = float((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))


    def _OPT50019(self, rqname, trcode):

        try:
            # 시가, 고가, 저가, 종가
            self.dollar_open = float((self._comm_get_data(trcode, "", rqname, 0, "시가")).lstrip('+-'))
            self.dollar_high = float((self._comm_get_data(trcode, "", rqname, 0, "고가")).lstrip('+-'))
            self.dollar_low = float((self._comm_get_data(trcode, "", rqname, 0, "저가")).lstrip('+-'))
            self.dollar_close = float((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))
        except:
            print("달러 다운로드 실패")

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

    def reset_dollar(self):
        self.dollar_open = ''
        self.dollar_high = ''
        self.dollar_low = ''
        self.dollar_close = ''

    def get_server_gubun(self):
        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()