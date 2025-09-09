# -*- coding:utf-8 -*-

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
import logging
import pandas_datareader.data as web
from datetime import date
from openpyxl import load_workbook
#import 고저점Trader as trader

form_class = uic.loadUiType("futureTrader.ui")[0]
file_path = "kospi200f_11_60M.csv"
log_path = "log/futureTrader60M.log"
result_path = "model3/pred_83_results.csv"

class MyWindow(QMainWindow, form_class):
    def __init__(self, gubun):
        super().__init__()

        self.setupUi(self)

        logging.basicConfig(filename='futureTrader60M.log', level=logging.DEBUG)

        self.kiwoom = Kiwoom()
        # login 화면 연결
        self.kiwoom.comm_connect()

        accouns_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
        accounts = self.kiwoom.get_login_info("ACCNO")

        accounts_list = accounts.split(';')[0:accouns_num]
        self.acc_no = accounts_list[1]
        self.comboBox.addItems(accounts_list)

        self.loss_cut = self.lineEdit.text()
        self.profit_real = self.lineEdit_3.text()

        futureCodes = self.kiwoom.dynamicCall("GetFutureList()")
        self.futureCode = futureCodes.split(";")[0]

        # 0: 수동 주문  1: 자동 주문
        self.auto_gubun = 0
        if int(gubun) == 1:
            self.start_auto()

        self.pushButton.clicked.connect(self.send_order)

        self.pushButton_2.clicked.connect(self.check_balance)
        self.pushButton_3.clicked.connect(self.start_auto)
        self.pushButton_4.clicked.connect(self.stop_auto)

        # Timer1
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)

        # Timer2
        self.timer2 = QTimer(self)
        self.timer2.start(1000 * 60)
        self.timer2.timeout.connect(self.timeout2)

    def timeout(self):

        current_time = QTime.currentTime()

        if current_time > QTime(15, 45, 0):
            print('장 종료..... 프로그램 종료')
            exit(0)

        trading_state = ""
        if self.auto_gubun == 1:
             trading_state = "자동 거래 실행 중"
        elif self.auto_gubun == 0:
            trading_state = "수동 주문 대기 중"
        else:
            trading_state = "주문시간 9:00 ~ 15:30"

        text_time = current_time.toString("hh:mm:ss")
        time_msg = "현재시간: " + text_time

        state = self.kiwoom.get_connect_state()
        if state == 1:
            state_msg = "서버 연결 중"
        else:
            state_msg = "서버 미 연결 중"

        self.statusbar.showMessage(state_msg + " | " + time_msg + "|" + trading_state)

    def timeout2(self):

        self.check_balance()

        # 조건 만족시 청산
        if self.auto_gubun: self.auto_liquidation()

        # 현재 60분봉 데이터(가격, 거래량, 미결제약정) 불러와서 매시간 1분 경과후 csv 파일에 저장, 예측 결과에 따라 주문
        now = datetime.datetime.now()
        m = now.minute
        h = now.hour
        if (h >= 9 and h <= 15 and m == 1) or (h == 15 and m == 31):
            self.update_data(h)
            # 고저점 predict, 결과 저장
            os.system("start cmd /c \"C:\\Users\\user\\Anaconda3\\envs\\test\\python 고저점Trader3.py " + "0 " + self.kiwoom.che_time + "\"")
            time.sleep(0.5)

            while True:
                try:
                    results_df = pd.read_csv(result_path, encoding='euc-kr')
                except:
                    print('pred_results.csv file read error...')
                    continue
                if results_df.loc[results_df.index.max(), 'date'] == self.kiwoom.che_time:
                    break
                if (datetime.datetime.now() - now).seconds > 60 * 10:
                    print('예측 프로세스 실패... 시간 경과 10분... 프로그램 종료')
                    logging.error('예측 프로세스 실패... 시간 경과 10분... 프로그램 종료')
                    exit(0)
            print(datetime.datetime.now())
            print('예측 완료,,, 결과 파일에 저장 확인...')
            logging.info('예측 완료,,, 결과 파일에 저장 확인...')

            if self.auto_gubun : self.new_order(results_df.loc[results_df.index.max()].values)

    def update_data(self, h):
        # 선물 시가, 고가, 저가, 현재가(종가), 거래량
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        if h == 15:
            self.kiwoom.set_input_value("시간단위", "30")
        else:
            self.kiwoom.set_input_value("시간단위", "60")
        self.kiwoom.comm_rq_data("가격거래량_req", "opt50029", "0", "3001")
        time.sleep(0.2)

        print('가격, 거래량 조회 완료...')

        # 선물 미결제 약정
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        if h == 15:
            self.kiwoom.set_input_value("시간단위", "30")
        else:
            self.kiwoom.set_input_value("시간단위", "60")
        self.kiwoom.comm_rq_data("미결제약정_req", "opt50062", "0", "3002")
        time.sleep(0.2)

        print('미결제 약정 조회 완료...')

        df = pd.read_csv(file_path, encoding='euc-kr')

        if self.kiwoom.che_time <= df.loc[df.index.max(), 'date']:
            return

        m = df.index.max()
        avg5 = (df.loc[m-3:, '종가'].sum() + float(self.kiwoom.close)) / 5
        avg20 = (df.loc[m-18:, '종가'].sum() + float(self.kiwoom.close)) / 20
        avg60 = (df.loc[m-58:, '종가'].sum() + float(self.kiwoom.close)) / 60
        avg120 = (df.loc[m-118:, '종가'].sum() + float(self.kiwoom.close)) / 120
        avg200 = (df.loc[m-198:, '종가'].sum() + float(self.kiwoom.close)) / 200

        new_data = {'date': self.kiwoom.che_time, '시가': self.kiwoom.open, '고가': self.kiwoom.high, '저가': self.kiwoom.low, '종가': self.kiwoom.close,
                    '5': str(avg5), '20': str(avg20), '60': str(avg60), '120': str(avg120), '200': str(avg200), '거래량': str(self.kiwoom.volumn), '미결제': self.kiwoom.open_interest}

        new_df = df.append(new_data, ignore_index=True)

        new_df.to_csv(file_path, index=False, encoding='euc-kr')

        print(datetime.datetime.now())
        logging.info(datetime.datetime.now())
        print("new data appended....")
        print(new_data)
        logging.info(new_data)

    def send_order(self):
        order_type_lookup = {'신규매매': 1, '정정': 2, '취소': 3}
        order_gubun_lookup = {'매도' : "1", '매수' : "2"}
        hoga_lookup = {'지정가': "1", '시장가': "3"}
        account = self.comboBox.currentText()
        order_type = self.comboBox_2.currentText()
        order_gubun = self.comboBox_4.currentText()
        code = self.lineEdit_2.text()
        hoga = self.comboBox_3.currentText()
        num = self.spinBox.value()
        price = str(self.spinBox_2.value())

        self.kiwoom.send_order("send_order_req", "0101", account, code,
                               order_type_lookup[order_type], order_gubun_lookup[order_gubun],
                               hoga_lookup[hoga], num, price, "")


    def check_balance(self):
        self.kiwoom.reset_opw20007_output()
        account_number = self.kiwoom.get_login_info("ACCNO")
        account_number = account_number.split(';')[1]

        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account_number)
            self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 2, "2000")

        # 주문가능총액
        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("OPW20010_req", "OPW20010", 0, "1000")
        time.sleep(0.2)

        # balance
        item = QTableWidgetItem(self.kiwoom.change_format(self.kiwoom.d2_deposit))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 0, item)

        for i in range(1, 2):
            item = QTableWidgetItem(self.kiwoom.opw20007_output['single'][i - 1])
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.tableWidget.setItem(0, i, item)

        self.tableWidget.resizeRowsToContents()

        # Item list
        item_count = len(self.kiwoom.opw20007_output['multi'])
        self.tableWidget_2.setRowCount(item_count)

        for j in range(item_count):
            item = QTableWidgetItem(self.kiwoom.opw20007_output['multi'][j][1])
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.tableWidget_2.setItem(j, 0, item)
            row = self.kiwoom.opw20007_output['multi'][j][3:]
            for i in range(len(row)):
                item = QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.tableWidget_2.setItem(j, i+1, item)

        self.tableWidget_2.resizeRowsToContents()

    def stop_auto(self):
        self.auto_gubun = 0

    def start_auto(self):
        #market_start_time = QTime(9, 0, 0)
        #market_end_time = QTime(15, 45, 0)
        #current_time = QTime.currentTime()

        #if market_start_time > current_time:
        #    print('시장 개시 전....')
        #    return
        #elif market_end_time < current_time:
        #    print('시장 종료....')
        #    return

        self.auto_gubun = 1

    def auto_liquidation(self):
        market_start_time = QTime(9, 0, 0)
        market_end_time = QTime(15, 45, 0)
        current_time = QTime.currentTime()

        if current_time < market_start_time or current_time > market_end_time:
            print("거래 가능 시간 아님....")
            return

        # 청산 가능 수량 조회
        now = datetime.datetime.now()
        self.kiwoom.reset_opw20007_output()
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        # 데이터 안 들어왔으면 1번 더 반복
        if not self.kiwoom.opw20007_output['single']:
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
            time.sleep(0.2)

        # 잔고 조회 실패시 청산 process 종료하고 return
        if not self.kiwoom.opw20007_output['single']:
            print("잔고 조회 실패, 청산 process return ...")
            return

        # 청산 조건: 약정 금약 > 0 and (약정후 4시간 55분 경과 or 손절가 이하로 손실)
        # 약정 금액 없으면 15시 5분에 프로그램종료

        contract_amt = int(self.kiwoom.opw20007_output['single'][0].replace(',', ''))

        if contract_amt == 0:
            print("청산할 잔고 없음....")
            #if current_time > QTime(15, 5, 0):
            #    print("오후 3시 5분 경과")
            #    print("프로그램 종료...")
            #    exit(0)
            return

        now = datetime.datetime.now()
        trans_df = pd.read_csv("trans.csv", encoding='euc-kr')
        trans_date = trans_df.loc[max(trans_df.index), '거래일시']
        trans_date = datetime.datetime.strptime(trans_date, '%Y/%m/%d/%H:%M')

        liquidation_condition = False
        if (now.day == trans_date.day and (now - trans_date).seconds >=17700) or \
           (now.day - trans_date.day > 0 and ((15 - trans_date.hour + 1 + now.hour - 9)*3600 + now.minute * 60 >= 17700)) or \
           (now.hour == 15 and (now - trans_date).seconds >=16200):
            print('청산 시간 만족')
            liquidation_condition = True
        else:
            print('청산 시간 부족')

        type = "청산 조건 아님"
        price = ""
        # 손절가 이하 또는 마지막 거래 이후 4시간 55분 경과후 무조건 청산
        for row in self.kiwoom.opw20007_output['multi']:
            # 매수 청산
            if row[2] == "2":
                if liquidation_condition or (float(row[5]) - float(row[4])) / float(row[4]) * 100 < float(self.loss_cut):
                    gubun = "1"
                    self.kiwoom.dynamicCall("SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                                 ["send_order_req", "0101", self.acc_no, row[0], 1, gubun, "3", int(row[3].replace(',', '')), "", ""])
                    type = "매수청산"
                    price = row[5]
            # 매도 청산
            else:
                if liquidation_condition or (float(row[4]) - float(row[5])) / float(row[4]) * 100 < float(self.loss_cut):
                    gubun = "2"
                    self.kiwoom.dynamicCall("SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                                 ["send_order_req", "0101", self.acc_no, row[0], 1, gubun, "3", int(row[3].replace(',', '')), "", ""])
                    type = "매도청산"
                    price = row[5]

        print(type)
        logging.info(type)

        # 체결확인 process

        if type == "매수청산":
            trans_df.loc[trans_df.loc[trans_df['매매구분'] == '매수'].index.max(), '청산일시'] = now.strftime("%Y/%m/%d/%H:%M")
            trans_df.loc[trans_df.loc[trans_df['매매구분'] == '매수'].index.max(), '청산가격'] = price
        elif type == '매도청산':
            trans_df.loc[trans_df.loc[trans_df['매매구분'] == '매도'].index.max(), '청산일시'] = now.strftime("%Y/%m/%d/%H:%M")
            trans_df.loc[trans_df.loc[trans_df['매매구분'] == '매도'].index.max(), '청산가격'] = price
        else:
            return

        trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')

        print('청산내역 trnas.csv에 저장')

        logging.info(datetime.datetime.now())
        logging.info('청산내역 trnas.csv에 저장')

    def new_order(self, pred):
        print('신규 주문 프로세스 시작...')
        now = datetime.datetime.now()
        logging.info(now)

        pred_date = pred[0]
        gubun = int(pred[1])

        if gubun == 0:
            print('신규 매수, 매도 없음...')
            return

        # 잔고 조회후 약정 금액 없으면 신규 주문
        self.kiwoom.reset_opw20007_output()
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        # 잔고 조회가 완료될 때 까지 3번 반복...
        repeat_no = 0
        while not self.kiwoom.opw20007_output['single']:
            if repeat_no > 2:
                print('주문 전 잔고 조회 실패 .... ')
                return
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
            time.sleep(0.2)
            repeat_no += 1

        # 신규 매매구분이 보유 매매구분과 같을 때 (예: 신규 매매구분=매수, 잔고 매매구분=매수) 거래 시간만 새로 저장
        if gubun == 1: m_gubun = '매도'
        else: m_gubun = '매수'

        if int(self.kiwoom.opw20007_output['single'][0].replace(',', '')) > 0 and int(self.kiwoom.opw20007_output['multi'][0][2]) == gubun:
            #trans_df = pd.read_csv('trans.csv', encoding='euc-kr')
            #trans_data = {'거래일시': self.kiwoom.che_time, '매매구분': m_gubun, '수량': self.kiwoom.opw20007_output['multi'][0][3],
            #              '거래가격': self.kiwoom.opw20007_output['multi'][0][4], '청산일시': '0', '청산가격': '0'}
            #trans_df = trans_df.append(trans_data, ignore_index=True)
            #trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')
            print('같은 position의 약정 금액이 존재.')
            return

        # 주문가능 수량
        self.kiwoom.new_quantity = ''
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("매도매수구분", str(gubun))
        self.kiwoom.comm_rq_data("OPW20009_req", "OPW20009", 0, "4000")
        time.sleep(0.2)

        # 주문 가능 수량 조회가 완료될 때 까지 3번 반복...
        repeat_no = 0
        while not self.kiwoom.new_quantity:
            if repeat_no > 2:
                print('주문 가능 수량 조회 실패 .... 시간 경과')
                logging.error('주문 가능 수량 조회 살패 .... 시간 경과')
                return
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.set_input_value("종목코드", self.futureCode)
            self.kiwoom.set_input_value("매도매수구분", str(gubun))
            self.kiwoom.comm_rq_data("OPW20009_req", "OPW20009", 0, "4000")
            time.sleep(0.2)
            repeat_no += 1

        # 주문 가능 수량 = 청산 가능 수량 + 신규 가능 수량
        if int(self.kiwoom.opw20007_output['single'][0].replace(',', '')) > 0:
            num = int(self.kiwoom.new_quantity) + int(self.kiwoom.opw20007_output['multi'][0][3])
        else:
            num = int(self.kiwoom.new_quantity / 2)

        if num < 1:
            print('주문 가능 금액 부족')
            logging.error('주문 가능 금액 부족')
            return

        # 시장가("3")로 주문
        if gubun == 1: # sell order
            price = ""
            self.kiwoom.dynamicCall(
                "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                ["send_order_req", "0101", self.acc_no, self.futureCode, 1, "1", "3", num, price, ""])
            print('매도주문 완료')
            logging.info('매도주문 완료')
        elif gubun == 2: # buy order
            price = ""
            self.kiwoom.dynamicCall(
                "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                ["send_order_req", "0101", self.acc_no, self.futureCode, 1, "2", "3", num, price, ""])
            print('매수주문 완료')
            logging.info('매수주문 완료')
        time.sleep(0.2)

        """
        # 잔고 조회후 거래 확인
        self.kiwoom.reset_opw20007_output()
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        # 거래 내역 잔고에 반영될 때 까지 3번 반복...
        repeat_no = 0
        while int(self.kiwoom.opw20007_output['single'][0].replace(',', '')) == 0:
            if repeat_no > 2:
                print('신규 주문 실패...trans.csv update 실패...')
                logging.error('신규 주문 실패...trans.csv update 실패...')
                return
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
            time.sleep(0.2)
            repeat_no += 1
        """

        # 주문 내역 거래내역 파일 (trans.csv) 에 저장
        trans_df = pd.read_csv('trans.csv', encoding='euc-kr')
        trans_data = {'거래일시': pred_date, '매매구분': m_gubun, '수량': num,
                      '거래가격': self.kiwoom.close, '청산일시': '0', '청산가격': '0'}
        trans_df = trans_df.append(trans_data, ignore_index=True)
        trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')
        print(trans_data)

        # 주문 내역 log 파일에 저장
        logging.info("주문내역")
        logging.info(trans_data)
        now = datetime.datetime.now()
        logging.info(now)

        print('신규 ' + m_gubun + ' 주문 완료... trans.csv 파일에 주문 내역 저장...')


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
        elif rqname == "OPW20007_req": # 선옵잔고현황 정산가 기준 합계
            self._OPW20007(rqname, trcode)
        elif rqname == "OPW20010_req": # 주문가능총액
            self._OPW20010(rqname, trcode)
        elif rqname == "가격거래량_req":
            self._OPT50029(rqname, trcode)
        elif rqname == "미결제약정_req":
            self._OPT50062(rqname, trcode)
        elif rqname == 'OPW20002_req':
            self._OPW20002(rqname, trcode)
        elif rqname == 'OPW20009_req':
            self._OPW20009(rqname, trcode)
        elif rqname == 'OPW20005_req':
            self._OPW20005(rqname, trcode)


        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _OPW20005(self, rqname, trcode):
        self.출력건수 = self._comm_get_data(trcode, "", rqname, 0, "출력건수")
        self.매매구분 = self._comm_get_data(trcode, "", rqname, 0, "매매구분")
        self.주문수량 = int(self._comm_get_data(trcode, "", rqname, 0, "주문수량"))
        self.체결수량 = int(self._comm_get_data(trcode, "", rqname, 0, "체결수량"))
        self.미체결수량 = int(self._comm_get_data(trcode, "", rqname, 0, "미체결수량"))
        self.체결가 = float(self._comm_get_data(trcode, "", rqname, 0, "체결가"))/100
        self.약정시간 = self._comm_get_data(trcode, "", rqname, 0, "약정시간")

    def _OPW20009(self, rqname, trcode):
        self.new_quantity = int(self._comm_get_data(trcode, "", rqname, 0, "신규가능수량"))
        self.liquid_quantity = int(self._comm_get_data(trcode, "", rqname, 0, "청산가능수량"))
        self.total_quantity = int(self._comm_get_data(trcode, "", rqname, 0, "총가능수량"))


    def _OPW20002(self, rqname, trcode):
        self.fee = int(self._comm_get_data(trcode, "", rqname, 0, "선물수수료"))
        self.sell_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물매도수량"))
        self.sell_price = int(self._comm_get_data(trcode, "", rqname, 0, "선물매도평균가격").lstrip('-'))*0.01
        self.buy_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물매수수량"))
        self.buy_price = int(self._comm_get_data(trcode, "", rqname, 0, "선물매수평균가격").lstrip('-'))*0.01
        self.resale_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물전매수량"))
        self.redemption_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물환매수량"))


    def _OPT50029(self, rqname, trcode):
        # 시가, 고가, 저가, 종가, 거래량
        self.close = (self._comm_get_data(trcode, "", rqname, 1, "현재가")).lstrip('+-')
        self.open = (self._comm_get_data(trcode, "", rqname, 1, "시가")).lstrip('+-')
        self.high = (self._comm_get_data(trcode, "", rqname, 1, "고가")).lstrip('+-')
        self.low = (self._comm_get_data(trcode, "", rqname, 1, "저가")).lstrip('+-')
        self.volumn = self._comm_get_data(trcode, "", rqname, 1, "거래량").replace(',', '')

        t = str(self._comm_get_data(trcode, "", rqname, 1, "체결시간"))
        self.che_time = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

    def _OPT50062(self, rqname, trcode):
        self.open_interest = self._comm_get_data(trcode, "", rqname, 0, "미결제약정")

    def _OPW20010(self, rqname, trcode):
        self.d2_deposit = self._comm_get_data(trcode, "", rqname, 0, "주문가능총액")

    def _OPW20007(self, rqname, trcode):
        self.reset_opw20007_output()

        # single data
        total_purchase_price = self._comm_get_data(trcode, "", rqname, 0, "약정금액합계")
        total_eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, 0, "평가손익금액")
        total_cnt = self._comm_get_data(trcode, "", rqname, 0, "출력건수")

        self.opw20007_output['single'].append(Kiwoom.change_format(total_purchase_price))
        self.opw20007_output['single'].append(Kiwoom.change_format(total_eval_profit_loss_price))
        self.opw20007_output['single'].append(Kiwoom.change_format(total_cnt))


        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        for i in range(rows):
            code = self._comm_get_data(trcode, "", rqname, i, "종목코드")
            name = self._comm_get_data(trcode, "", rqname, i, "종목명")
            gubun = self._comm_get_data(trcode, "", rqname, i, "매도매수구분")
            quantity = self._comm_get_data(trcode, "", rqname, i, "수량")
            purchase_price = self._comm_get_data(trcode, "", rqname, i, "매입단가")
            current_price = self._comm_get_data(trcode, "", rqname, i, "현재가")
            eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, i, "평가손익")
            if gubun == "2":
                earning_rate = (float(current_price) - float(purchase_price)) / float(purchase_price) * 100
            else:
                earning_rate = (float(purchase_price) - float(current_price)) / float(purchase_price) * 100
            quantity = Kiwoom.change_format(quantity)
            purchase_price = str(float(purchase_price) / 100)
            current_price = str(float(current_price) / 100)
            eval_profit_loss_price = Kiwoom.change_format(eval_profit_loss_price)
            earning_rate = str(earning_rate)

            self.opw20007_output['multi'].append([code, name, gubun, quantity, purchase_price, current_price,
                                                  eval_profit_loss_price, earning_rate])


    def send_order(self, rqname, screen_no, acc_no, code, order_type, gubun, hoga, quantity, price, order_no):
        self.dynamicCall("SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                         [rqname, screen_no, acc_no, code, order_type, gubun,  hoga, quantity, price, order_no])

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

    def reset_opw20007_output(self):
        self.opw20007_output = {'single': [], 'multi': []}

    def reset_sell_output(self):
        self.sell_output = {'single': [], 'multi': []}

    def get_server_gubun(self):
        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow(sys.argv[1])
    myWindow.show()
    app.exec_()