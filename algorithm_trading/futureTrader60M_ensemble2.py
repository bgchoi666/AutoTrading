# Copyright 2023 Bimghi Choi. All Rights Reserved.
# 3개의 서로 다른 알고리즘들의 거래 결과를 평균해서 최종 거래 

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
import save_trans as st

form_class = uic.loadUiType("futureTrader.ui")[0]

file_path = "kospi200f_11_60M.csv"#version 2
v4_path = "../알고리즘트레이딩4/kospi200f_11_60M.csv"#version 4
new_path = "../알고리즘트레이딩_new/kospi200f_11_60M.csv"#version 'new'
알고리즘1D_path = "../알고리즘1D/kospi200f_1D.csv"#일봉 데이터
version = 'new'

log_path = "futureTrader60M.log"
result_path = "pred_83_results.csv"

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

        for self.acc_no in accounts_list:
            if self.acc_no[8:] == '31':
                break

        self.comboBox.addItems(accounts_list)

        self.loss_cut = self.lineEdit.text()
        self.profit_real = self.lineEdit_3.text()

        futureCodes = self.kiwoom.dynamicCall("GetFutureList()")
        self.futureCode = futureCodes.split(";")[0]
        self.lineEdit_2.setText(str(self.futureCode))

        # portfolio 구성 요소, 2: 알고리즘2, 40:알고리즘4, 41: 알고리즘4-1, 42:알고리즘4-2, 'new':알고리즘_new
        self.port = [20]

        # 삼성전자, 카카오 데이터를 받기 위하여
        self.samsungCode = '005930'
        self.kakaoCode = '035720'
        self.dollarCode = '175V1000'

        self.samsung_path = "F:/키움_주식_달러선물/samsung_60M.csv"
        self.kakao_path = "F:/키움_주식_달러선물/kakao_60M.csv"
        self.dollar_path = "../dollar_1D/dollar_1D.csv"
        self.kakao_1D_path = "../stock_1D/kakao_1D.csv"
        self.samsung_1D_path = "../stock_1D/samsung_1D.csv"

        # 0: 수동 주문  1: 자동 주문
        self.auto_gubun = 0
        if int(gubun) == 1:
            self.start_auto()

        self.pushButton.clicked.connect(self.send_order)

        self.pushButton_2.clicked.connect(self.check_balance)
        self.pushButton_3.clicked.connect(self.start_auto)
        self.pushButton_4.clicked.connect(self.stop_auto)
        self.pushButton_5.clicked.connect(self.profit_graph)

        self.msg = ""

        # Timer1
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)

        # Timer2
        self.timer2 = QTimer(self)
        self.timer2.start(1000 * 60)
        self.timer2.timeout.connect(self.timeout2)

        # 장 종료후 단일가 거래 실시 여부
        self.clear = False

        #장 종료 후 청산 가격 일치 여부
        self.청산가격일치 = False

    def timeout(self):

        if QTime.currentTime() > QTime(15, 50, 0):
            print("거래시간 종료")
            logging.info("거래시간 종료")
            logging.info(datetime.datetime.now())
            exit(0)

        current_time = QTime.currentTime()

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

        try:
            self.cnt_per_port = self.check_balance()
        except:
            print("check balance error!!")
            logging.info("check balance error!!")

        #거래 가능 시간 check
        if QTime.currentTime() > QTime(15, 47, 0) or QTime.currentTime() < QTime(9, 5, 0):

            # 일 마감 후 청산가격 조정
            if not self.청산가격일치 and QTime.currentTime() > QTime(15, 47, 0):
                # 현재가 정보
                self.kiwoom.current_price = 0
                self.kiwoom.set_input_value("종목코드", self.futureCode)
                self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
                time.sleep(0.2)

                # 현재가 조회가 완료될 때 까지 10번 반복...
                cnt = 0
                while self.kiwoom.current_price == 0:
                    time.sleep(2)
                    if cnt > 10:
                        print('주문 후 10번 현재가 조회 실패')
                        logging.info('주문 후 10번 현재가 조회 실패')
                        return
                    self.kiwoom.set_input_value("종목코드", self.futureCode)
                    self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
                    cnt += 1

                t_df = pd.read_csv("trans.csv", encoding='euc-kr')
                if t_df['청산일시'].values[-1] > datetime.datetime.now().strftime("%Y/%m/%d/15:35"):
                    t_df.loc[len(t_df) - 1, '청산가격'] = self.kiwoom.current_price

                    t_df.to_csv("trans.csv", index=False,  encoding='euc-kr')

                if QTime.currentTime() > QTime(15, 47, 0):
                    self.청산가격일치 = True

            return

        # 조건 만족시 청산
        if self.auto_gubun and QTime.currentTime() < QTime(15, 35, 0) and QTime.currentTime() > QTime(8, 45, 0):
            try:
                r = self.auto_liquidation(False)
            except:
                print("auto liquidation(False) error!!")
                logging.info("auto liquidation(False) error!!")
            if not r:
                print("loss-cut으로 인한 종료")
                logging.info("loss-cut으로 인한 종료")
                exit(0)

        # 현재 60분봉 데이터(가격, 거래량, 미결제약정) 불러와서 매시간 1분 경과후 csv 파일에 저장, 예측 결과에 따라 주문
        now = datetime.datetime.now()
        m = now.minute
        h = now.hour

        # non overnight, 장 졸료후 단일가 거래로 청산
        if not self.clear and h == 15 and m > 35 and m < 45:
            self.auto_liquidation(True)
            self.clear = True

        # 달러, 주식 데이터 다운로드, 거래
        if h == 15 and m >= 46:
            # 달러 선물 update, trade
            self.dollar_update_trade(self.dollarCode, self.dollar_path)

            # kakao update, trade
            self.kakao_update_trade(self.kakaoCode, self.kakao_1D_path)

            # samsung update, trade
            self.samsung_update_trade(self.samsungCode, self.samsung_1D_path)

            # samsung update, trade
            self.알고리즘1D_update()

            # send the trading signals to e-mail
            self.send_result_to_email(self.msg)

            return

        elif m != 45:
            return

        self.update_data()
        self.update_v4_data()
        shutil.copy(v4_path, new_path)

        if h == 15:
            return

        n = self.calc_order_qty()

        # 예측 프로세스 에러로 인한 데이터 재 다운로드 후 시도
        if n == 'error':
            self.update_data()
            self.update_v4_data()
            shutil.copy(v4_path, new_path)

            n = self.calc_order_qty()

            # 계속 되는 에러 프로그램 종료
            if n == 'error':
                print("예측 프로세스 연속 에러 .... 프로그램 종료")
                logging.info("예측 프로세스 연속 에러 .... 프로그램 종료")
                exit(0)

        if self.auto_gubun:
            if n > 0:
                self.new_order(2, n)
            elif n < 0:
                self.new_order(1, abs(n))
            else:
                print("거래 없음")

    # 각각의 앙상블 예측 값 산출
    def predict(self, version):
        if version == 20:
            # 알고리즘트레이딩 version2 ensemble predict
            os.system("start cmd /c \"C:\\Users\\user\\Anaconda3\\envs\\test\\python ensemble.py \"")
            result_path = "../알고리즘트레이딩2/pred_83_results.csv"
        elif version == 21:
            # 알고리즘트레이딩 version4 ensemble predict
            os.system("start cmd /c \"C:\\Users\\user\\Anaconda3\\envs\\test\\python ensemble1.py \"")
            result_path = "../알고리즘트레이딩2/pred_83_results.csv"
        elif version == 22:
            # 알고리즘트레이딩 version4 ensemble predict
            os.system("start cmd /c \"C:\\Users\\user\\Anaconda3\\envs\\test\\python ensemble2.py \"")
            result_path = "../알고리즘트레이딩2/pred_83_results.csv"
        elif version == 40:
            # 알고리즘트레이딩 version4 ensemble predict
            os.system("start cmd /c \"version4_ensemble.bat\"")
            result_path = "../알고리즘트레이딩4/pred_88_results.csv"
        elif version == 41:
            # 알고리즘트레이딩 version4 ensemble predict
            os.system("start cmd /c \"version4_ensemble1.bat\"")
            result_path = "../알고리즘트레이딩4/pred_88_results.csv"
        elif version == 42:
            # 알고리즘트레이딩 version4 ensemble predict
            os.system("start cmd /c \"version4_ensemble2.bat\"")
            result_path = "../알고리즘트레이딩4/pred_88_results.csv"
        elif version == 'new0':
            # 알고리즘트레이딩 version new ensemble predict
            os.system("start cmd /c \"new_ensemble.bat\"")
            result_path = "../알고리즘트레이딩_new/pred_88_results.csv"
        elif version == 'new1':
            # 알고리즘트레이딩 version new ensemble predict
            os.system("start cmd /c \"new_ensemble1.bat\"")
            result_path = "../알고리즘트레이딩_new/pred_88_results.csv"
        elif version == 'new2':
            # 알고리즘트레이딩 version new ensemble predict
            os.system("start cmd /c \"new_ensemble2.bat\"")
            result_path = "../알고리즘트레이딩_new/pred_88_results.csv"

        else:
            print("version error!! ... exit")
            logging.info("version error!! ... exit")
            exit(0)

        time.sleep(0.5)

        now = datetime.datetime.now()

        if now.weekday() == 5:
            return 0
        elif now.weekday() == 6:
            return 0
        elif (now.hour == 15 and now.minute > 45) or now.hour > 15:
            return 0
        elif now.hour < 9 and now.minute < 45:
            return 0
        else:
            chetime = now.strftime("%Y/%m/%d/%H:00")


        while True:
            try:
                results_df = pd.read_csv(result_path, encoding='euc-kr')
            except:
                print('pred_results.csv file read error...')
                continue
            if results_df.loc[results_df.index.max(), 'date'] == chetime:
                break
            if (datetime.datetime.now() - now).seconds > 60 :
                print('알고리즘 version ' + str(version))
                print('예측 프로세스 실패... 시간 경과 1분', chetime)
                logging.info('알고리즘 version ' + str(version))
                logging.error('예측 프로세스 실패... 시간 경과 1분' + chetime)
                return -1
            time.sleep(0.5)

        v = results_df.loc[results_df.index.max()].values

        r = int(v[1])

        if version != 20 and version != 21 and version != 22: #알고리즘2는 target이 2, 다른것은 3, 조정 필요
            r = int(v[1]) + 1

        return r

    # 각 앙상블의 현재 보유 수량 및 메수/매도 상태 출력
    def show_state(self, version):

        # 각 version의 tran file 및 통합 tran file read
        if version == self.port[0]:
            trans_df = pd.read_csv("trans_"+str(self.port[0])+".csv", encoding="euc-kr")
        elif version == self.port[1]:
            trans_df = pd.read_csv("trans_"+str(self.port[1])+".csv", encoding="euc-kr")
        elif version == self.port[2]:
            trans_df = pd.read_csv("trans_"+str(self.port[2])+".csv", encoding="euc-kr")
        else:
            print("version error", version)
            exit(0)


        # 오늘 잔고 수량과 매수매도 상태
        now = datetime.datetime.now()
        trans_df = trans_df.loc[trans_df["거래일시"] >= now.strftime("%Y/%m/%d/09:00")]

        num = trans_df.loc[trans_df["매매구분"] == "매수", "수량"].values.sum() - \
              trans_df.loc[trans_df["매매구분"] == "매도", "수량"].values.sum()

        return num, trans_df

    def calc_order_qty(self):
        """
        # 주문가능총액, 증거금총액 조회
        self.kiwoom.d2_deposit = ''
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20010_req", "OPW20010", 0, "1000")
        time.sleep(0.2)

        repeat_no = 0
        while not self.kiwoom.d2_deposit:
            if repeat_no > 10:
                print("balance 조회 실패")
                logging.info("balance 조회 실패")
                return 0
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.comm_rq_data("OPW20010_req", "OPW20010", 0, "1000")
            time.sleep(0.2)
            repeat_no += 1
        """

        # 현재가 정보
        self.kiwoom.current_price = 0
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
        time.sleep(0.2)

        # 각 trans file에 거래내역 저장, 전체 거래량 계산
        total_qty = 0
        trading_date = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

        if self.kiwoom.current_price != 0:
            for v in self.port:

                # 기존 temp file 제거
                if os.path.isfile('trans_' + str(v) + '_temp.csv'):
                    os.remove('trans_' + str(v) + '_temp.csv')

                pred = self.predict(v)
                if pred < 0: # 예측 프로세스 에러
                    return 'error'
                num, df = self.show_state(v)

                # 진입후 4시간 경과시 무조건 청산
                #if self.check_overtime(df) and num != 0:
                #    if num > 0:
                #        df = df.append(
                #            {'거래일시': trading_date, '매매구분': '매도', '수량': num, '거래가격': 0, '청산일시': trading_date,
                #             '청산가격': self.kiwoom.current_price},
                #            ignore_index=True)
                #    else:
                #        df = df.append(
                #            {'거래일시': trading_date, '매매구분': '매수', '수량': -num, '거래가격': 0, '청산일시': trading_date,
                #             '청산가격': self.kiwoom.current_price},
                #            ignore_index=True)
                #
                #    df.to_csv('trans_' + str(v) + '_temp.csv', index=False, encoding='euc-kr')
                #    total_qty -= num

                if pred == 2 and num <= 0:
                    trading_num = -num + 1#self.cnt_per_port#-num * 2 + int(int(self.kiwoom.d2_deposit) / 1.35 / 0.075 / 250000 / self.kiwoom.current_price / 3)
                    df = df.append({'거래일시': trading_date, '매매구분': '매수', '수량': trading_num, '거래가격': self.kiwoom.current_price, '청산일시': '0', '청산가격': '0'},
                              ignore_index=True)
                    df.to_csv('trans_' + str(v) + '_temp.csv', index=False, encoding='euc-kr')
                    total_qty += trading_num
                elif pred == 1 and num >= 0:
                    trading_num = num + 1#self.cnt_per_port#num * 2 + int(int(self.kiwoom.d2_deposit) / 1.35 / 0.075 / 250000 / self.kiwoom.current_price / 3)
                    df = df.append({'거래일시': trading_date, '매매구분': '매도', '수량': trading_num, '거래가격': self.kiwoom.current_price, '청산일시': '0', '청산가격': '0'},
                              ignore_index=True)
                    df.to_csv('trans_' + str(v) + '_temp.csv', index=False, encoding='euc-kr')
                    total_qty -= trading_num

                print("version " + str(v) + "예측, 누적수량", pred, total_qty)
                logging.info("version " + str(v) + "예측: " + str(pred) + " 누적수량: " + str(total_qty))
        else:
            print("현재가 정보 조회 error!!!")
            logging.info("현재가 정보 조회 error!!!")
            return 'error'

        return int(total_qty)

    def check_overtime(self, trans_df):

        # 현재시간, 마지막 거래 일시 조회
        now = datetime.datetime.now()
        trans_date = trans_df.loc[max(trans_df.index), '거래일시']

        # 진입후 4시간 경과 여부 check, 15시인 경우 3시간 30분 경과 여부 check
        trans_date = datetime.datetime.strptime(trans_date, '%Y/%m/%d/%H:%M')
        if ((now.day == trans_date.day and (now - trans_date).seconds >=18000) or
           ((now.day - trans_date.day > 0 or now.month > trans_date.month or now.year > trans_date.year) and
           ((15 - trans_date.hour + 1 + now.hour - 9)*3600 + now.minute * 60 >= 18000))):
            return 1
        else:
            return 0

    def new_order(self, state, num):

        print('신규 주문 프로세스 시작...')
        print('예측 결과, 수량:', state, num)
        now = datetime.datetime.now()
        print(now)

        logging.info(now)
        logging.info('예측 결과: ' + str(state) + " " + str(num))

        # 주문가능수량 조회
        self.kiwoom.total_quantity = ''
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("매도수구분", str(state))
        self.kiwoom.set_input_value("주문유형", "3")
        self.kiwoom.comm_rq_data("OPW20009_req", "opw20009", 0, "7000")
        time.sleep(0.2)

        repeat_no = 0
        while not self.kiwoom.total_quantity:
            if repeat_no > 10:
                print("주문가능 수량 조회 실패")
                logging.info("주문가능 수량 조회 실패")
                return
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.set_input_value("종목코드", self.futureCode)
            self.kiwoom.set_input_value("매도수구분", str(state))
            self.kiwoom.set_input_value("주문유형", "3")
            self.kiwoom.comm_rq_data("OPW20009_req", "opw20009", 0, "7000")
            time.sleep(0.2)
            repeat_no += 1

        num0 = min(num, self.kiwoom.total_quantity)

        # 현재가 정보
        self.kiwoom.current_price = 0
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
        time.sleep(0.2)

        # 현재가 조회가 완료될 때 까지 10번 반복...
        now = datetime.datetime.now()
        cnt = 0
        while self.kiwoom.current_price == 0:
            time.sleep(2)
            if cnt > 10:
                print('주문 후 10번 현재가 조회 실패')
                logging.info('주문 후 10번 현재가 조회 실패')
                return
            self.kiwoom.set_input_value("종목코드", self.futureCode)
            self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
            cnt += 1

        # 시장가("3")로 주문
        if state == 1: # sell order
            self.kiwoom.dynamicCall(
                "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                ["send_order_req", "0101", self.acc_no, self.futureCode, 1, "1", "3", int(num0), "", ""])
            print('매도주문 완료')
            logging.info('매도주문 완료')
        elif state == 2: # buy order
            self.kiwoom.dynamicCall(
                "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                ["send_order_req", "0101", self.acc_no, self.futureCode, 1, "2", "3", int(num0), "", ""])
            print('매수주문 완료')
            logging.info('매수주문 완료')

        time.sleep(60)

        purchase_price = self.kiwoom.current_price
        purchase_num = num0

        if state == 1:
            gubun = "매도"
        elif state == 2:
            gubun = "매수"
        else:
            print("매매구분 error")
            logging.info("매매구분 error")
            return

        # 거래내역 trans file에 저장
        try:
            st.save_trans(gubun, purchase_num, purchase_price)
        except:
            print("saving new transaction error")
            logging.info("saving new transaction error")

        # temporary 개별 거래 파일 본 파일로 move
        if os.path.isfile('trans_20_temp.csv'): shutil.move('trans_20_temp.csv', 'trans_20.csv')
        if os.path.isfile('trans_21_temp.csv'): shutil.move('trans_21_temp.csv', 'trans_21.csv')
        if os.path.isfile('trans_22_temp.csv'): shutil.move('trans_22_temp.csv', 'trans_22.csv')
        if os.path.isfile('trans_40_temp.csv'): shutil.move('trans_40_temp.csv', 'trans_40.csv')
        if os.path.isfile('trans_41_temp.csv'): shutil.move('trans_41_temp.csv', 'trans_41.csv')
        if os.path.isfile('trans_42_temp.csv'): shutil.move('trans_42_temp.csv', 'trans_42.csv')
        if os.path.isfile('trans_new0_temp.csv'): shutil.move('trans_new0_temp.csv', 'trans_new0.csv')
        if os.path.isfile('trans_new1_temp.csv'): shutil.move('trans_new1_temp.csv', 'trans_new1.csv')
        if os.path.isfile('trans_new2_temp.csv'): shutil.move('trans_new2_temp.csv', 'trans_new2.csv')


        # 잔여 수량 거래
        if num0 < num:
            # 현재가 정보
            self.kiwoom.current_price = 0
            self.kiwoom.set_input_value("종목코드", self.futureCode)
            self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
            time.sleep(0.2)
    
            # 현재가 조회가 완료될 때 까지 10번 반복...
            now = datetime.datetime.now()
            cnt = 0
            while self.kiwoom.current_price == 0:
                time.sleep(2)
                if cnt > 10:
                    print('주문 후 10번 현재가 조회 실패')
                    return
                self.kiwoom.set_input_value("종목코드", self.futureCode)
                self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
                cnt += 1
    
            # 주문가능수량 조회
            self.kiwoom.total_quantity = ''
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.set_input_value("종목코드", self.futureCode)
            self.kiwoom.set_input_value("매도수구분", str(state))
            self.kiwoom.set_input_value("주문유형", "3")
            self.kiwoom.comm_rq_data("OPW20009_req", "opw20009", 0, "7000")
            time.sleep(0.2)
    
            repeat_no = 0
            while not self.kiwoom.total_quantity:
                time.sleep(2)
                if repeat_no > 10:
                    print("주문가능 수량 조회 실패")
                    return
                self.kiwoom.set_input_value("계좌번호", self.acc_no)
                self.kiwoom.set_input_value("종목코드", self.futureCode)
                self.kiwoom.set_input_value("매도수구분", str(state))
                self.kiwoom.set_input_value("주문유형", "3")
                self.kiwoom.comm_rq_data("OPW20009_req", "opw20009", 0, "7000")
                repeat_no += 1
    
            purchase_price = self.kiwoom.current_price
            purchase_num = min(num - num0, self.kiwoom.total_quantity) # 추가 거래 수량
    
            # 시장가("3")로 주문
            if state == 1: # sell order
                self.kiwoom.dynamicCall(
                    "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                    ["send_order_req", "0101", self.acc_no, self.futureCode, 1, "1", "3", int(purchase_num), "", ""])
                print('추가 매도주문 완료')
                logging.info('추가 매도주문 완료')
            elif state == 2: # buy order
                self.kiwoom.dynamicCall(
                    "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                    ["send_order_req", "0101", self.acc_no, self.futureCode, 1, "2", "3", int( purchase_num), "", ""])
                print('추가 매수주문 완료')
                logging.info('추가 매수주문 완료')
    
            if state == 1:
                gubun = "매도"
            elif state == 2:
                gubun = "매수"
            else:
                print("매매구분 error")
                return
    
            # 거래내역 trans file에 저장
            try:
                st.save_trans(gubun, purchase_num, purchase_price)
            except:
                print("saving tran_test.csv error!!")
            print(gubun + '추가 주문 완료... trans.csv 파일에 주문 내역 저장...')
            logging.info(gubun + '추가 주문 완료... trans.csv 파일에 주문 내역 저장...')
        

        print(gubun + ' 주문 완료... trans.csv 파일에 주문 내역 저장...')
        logging.info(gubun + ' 주문 완료... trans.csv 파일에 주문 내역 저장...')

    def auto_liquidation(self, liquidation_condition):
        market_start_time = QTime(9, 0, 0)
        market_end_time = QTime(15, 45, 0)
        current_time = QTime.currentTime()

        if current_time < market_start_time or current_time > market_end_time:
            print("거래 가능 시간 아님....")
            return False

        # 구입 가격, 현재가, 구입 수량 조회, 데이터 안 들어왔으면 10번 더 반복
        self.kiwoom.reset_opw20007_output()
        cnt = 0
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)
        while not self.kiwoom.opw20007_output['single']:
            if cnt > 10:
                print("잔고 조회 실패, 청산 process return ...")
                logging.info("잔고 조회 실패, 청산 process return ...")
                return False
            self.kiwoom.set_input_value("계좌번호", self.acc_no)
            self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
            time.sleep(0.2)
            cnt += 1


        # 청산 조건: 약정 금액 > 0
        contract_amt = int(self.kiwoom.opw20007_output['single'][0].replace(',', ''))

        if contract_amt == 0:
            print("청산할 잔고 없음....")
            return True

        code = ''
        current_gubun = ''
        holding_num = 0
        purchase_price = 0.0
        current_price = 0.0
        for row in self.kiwoom.opw20007_output['multi']:
            code = row[0]
            if code != self.futureCode:
                continue
            current_gubun = row[2]
            holding_num = int(row[3].replace(',', ''))
            purchase_price = float(row[4].replace(',', ''))
            current_price = float(row[5].replace(',', ''))
        if code != self.futureCode:
            print("청산할 잔고 없음....")
            return True

        청산 = False
        type = "청산 조건 아님"

        # 손절가 이하 또는 15시 35분 이후 무조건 청산
        self.loss_cut = self.lineEdit.text()

        # 매수 청산
        if current_gubun == "2":
            if liquidation_condition or (current_price - purchase_price) / float(
                    purchase_price) * 100 < float(self.loss_cut):
                gubun = "1"
                self.kiwoom.dynamicCall(
                    "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                    ["send_order_req", "0101", self.acc_no, code, 1, gubun, "3", holding_num, "", ""])
                type = "매수청산"
                청산 = True

        # 매도 청산
        else:
            if liquidation_condition or (purchase_price - current_price) / float(
                    purchase_price) * 100 < float(self.loss_cut):
                gubun = "2"
                self.kiwoom.dynamicCall(
                    "SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                    ["send_order_req", "0101", self.acc_no, code, 1, gubun, "3", holding_num, "", ""])
                type = "매도청산"
                청산 = True

        print(type)
        logging.info(type)

        now = datetime.datetime.now()
        trading_date = now.strftime("%Y/%m/%d/%H:%M")

        if 청산:
            # 청산 내역 통합 trans file에 저장
            try:
                st.save_liquidation(current_price)
            except:
                print("청산 내역 저장 error")
                logging.info("청산 내역 저장 error")

            # 청산 내역 개별 tran file에 저장
            for v in self.port:
                num, df = self.show_state(v)
                if num > 0:
                    df = df.append(
                        {'거래일시': trading_date, '매매구분': '매도', '수량': num, '거래가격': 0, '청산일시': trading_date,
                         '청산가격': current_price},
                        ignore_index=True)
                elif num < 0:
                    df = df.append(
                        {'거래일시': trading_date, '매매구분': '매수', '수량': -num, '거래가격': 0, '청산일시': trading_date,
                         '청산가격': current_price},
                        ignore_index=True)
                else:
                    continue

                df.to_csv("trans_" + str(v) + ".csv", index=False, encoding="euc-kr")

            print('청산내역 개별 tran_x.csv에 저장')

            logging.info(datetime.datetime.now())
            logging.info('청산내역 개별 trnas_x.csv에 저장')

            if trading_date[11:13] < '15':
                print("loss_cut으로 인한 프로그램 종료")
                logging.info("loss_cut으로 인한 프로그램 종료")
                return False

        return True

    def check_balance(self):
        #self.kiwoom.reset_opw20007_output()
        #account_number = self.kiwoom.get_login_info("ACCNO")
        #account_number = account_number.split(';')[1]
        account_number = self.acc_no

        # 현재가 정보
        self.kiwoom.current_price = 0
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.comm_rq_data("opt50001_req", "opt50001", 0, "6000")
        time.sleep(0.2)

        self.kiwoom.reset_opw20007_output()
        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        if not self.kiwoom.opw20007_output['single']:
            print("balance 조회 실패")
            logging.info("balance 조회 실패")
            return

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account_number)
            self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 2, "2000")

        # 주문가능총액, 증거금총액 조회
        self.kiwoom.d2_deposit = ''
        self.kiwoom.margin = ''
        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("OPW20010_req", "OPW20010", 0, "1000")
        time.sleep(0.2)

        if not self.kiwoom.d2_deposit:
            print("balance 조회 실패")
            logging.info("balance 조회 실패")
            return

        # balance
        item = QTableWidgetItem(self.kiwoom.change_format(self.kiwoom.d2_deposit))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 0, item)

        # 약정 금액 합계 출력
        item = QTableWidgetItem(self.kiwoom.opw20007_output['single'][0])
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 1, item)

        # 증거금 출력
        item = QTableWidgetItem(self.kiwoom.change_format(self.kiwoom.margin))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 2, item)

        # 추정예탁자산 계산 및 출력
        if len(self.kiwoom.opw20007_output['multi']) > 0:
            extimated_deposit_assets = int(self.kiwoom.d2_deposit) + int(self.kiwoom.margin) + \
                                   int(self.kiwoom.opw20007_output['multi'][0][6].replace(',', ''))
        else:
            extimated_deposit_assets = int(self.kiwoom.d2_deposit) + int(self.kiwoom.margin)

        item = QTableWidgetItem(str(extimated_deposit_assets))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 3, item)

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

        return round(extimated_deposit_assets / (float(self.kiwoom.current_price) * 250000 * 0.078) / 3, 2)

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

        file_path4 = "../알고리즘트레이딩4/kospi200f_11_60M.csv"
        df = pd.read_csv(file_path4, encoding='euc-kr')
        last_date = df.loc[df.index.max(), 'date']

        if QTime.currentTime() <= QTime(15, 45, 0) and self.kiwoom.che_time <= last_date:
            return
        elif QTime.currentTime() > QTime(15, 45, 0) and last_date[11:13] == '15' and self.kiwoom.che_time <= last_date:
            print("알고리즘트레이딩4 추가 데이터 없음")
            logging.info("알고리즘트레이딩4 추가 데이터 없음")
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
            logging.info("알고리즘_1D 추가 데이터 없음")
            return

        # 선물 시가, 고가, 저가, 현재가(종가), 거래량
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("기준일", "")
        self.kiwoom.comm_rq_data("일가격거래량_req", "opt50030", "0", "8000")
        time.sleep(0.2)

        print('가격, 거래량 조회 완료...')

        # 선물 미결제 약정
        self.kiwoom.set_input_value("종목코드", self.futureCode)
        self.kiwoom.set_input_value("시간단위", "1800")
        self.kiwoom.comm_rq_data("미결제약정_req", "opt50062", "0", "3002")
        time.sleep(0.2)

        print('미결제 약정 조회 완료...')

        columns = ['date', '시가', '고가', '저가', '종가', 'PDI', 'MDI', 'ADX', 'SlowK', 'SlowD', '거래량', '미결제']
        new_df = pd.DataFrame(columns=columns)

        new_data = {'date': today, '시가': self.kiwoom.open_1D, '고가': self.kiwoom.high_1D, '저가': self.kiwoom.low_1D,
                    '종가': self.kiwoom.close_1D, 'PDI': 0, 'MDI': 0, 'ADX': 0, 'SlowK': 0, 'SlowD': 0,
                    '거래량': self.kiwoom.volume_1D, '미결제': self.kiwoom.open_interest}

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
            print("알고리즘트레이딩2 추가 데이터 없음")
            logging.info("알고리즘트레이딩2 추가 데이터 없음")
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
            print("알고리즘트레이딩2 추가 데이터 없음")
            logging.info("알고리즘트레이딩2 추가 데이터 없음")
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

    """
    def stock_update_data(self, code, path):
        # 주식 시가, 고가, 저가, 현재가(종가), 거래량s
        self.kiwoom.reset_opt10080_output()
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("틱범위", "60")
        self.kiwoom.comm_rq_data("주식데이터_req", "opt10080", "0", "4001")
        time.sleep(0.2)

        if not self.kiwoom.opt10080_output['multi']:
            print(code + " update 실패")
            logging.info(code + " update 실패")
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
            print("추가 데이터 없음")
            return
        new_df = pd.DataFrame(np.array(new_data), columns=['date', '시가', '고가', '저가', '종가', '거래량'])
        new_df = pd.concat([df, new_df], axis=0, ignore_index=True)

        new_df.to_csv(path, index=False, encoding='euc-kr')

        print(code + ': 데이터 추가 완료...')
        logging.info(datetime.datetime.now())
        logging.info(code + ": data appended....")
    """

    def dollar_update_trade(self, code, path):

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # ========== 데이터 파일 update ===============================
        df = pd.read_csv(path, encoding='euc-kr')

        if df['date'].values[-1] >= today:
            print("달러 거래 종료")
            logging.info("달러 거래 종료")
            return

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

        # 데이터 추가
        new_data = {'date': today, '시가': self.kiwoom.dollar_open,
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


        # ==================== 앙상블 예측 ================================
        os.system("start cmd /c \"dollar_ensemble_predict.bat\"")
        time.sleep(0.5)

        now = datetime.datetime.now()

        while True:
            try:
                dollar_results_df = pd.read_csv("../dollar_1D/models/doller_1D_results.csv", encoding='euc-kr')
            except:
                print('pred_results.csv file read error...')
                if (datetime.datetime.now() - now).seconds > 60 * 5:
                    print('달러 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                    logging.error('달러 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                    return
                continue
            if dollar_results_df.loc[dollar_results_df.index.max(), 'dates'] == today:
                break
            if (datetime.datetime.now() - now).seconds > 60 * 5:
                print('달러 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                logging.error('달러 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                return

        # ==================== 거래 =======================================
        buy_sell_signal = int(dollar_results_df['results'].values[-1])
        if buy_sell_signal == 1:
            gubun = "매수"
            order_gubun = "2"
        else:
            gubun = "매도"
            order_gubun = "1"

        self.msg = "달러: " + gubun
        #self.send_result_to_email(gubun)

        # 주문 내역 거래내역 파일 (dollar_trans.csv) 에 저장
        trans_df = pd.read_csv('dollar_trans.csv', encoding='euc-kr')

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
            trans_df.to_csv('dollar_trans.csv', index=False, encoding='euc-kr')

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
            trans_df.to_csv('dollar_trans.csv', index=False, encoding='euc-kr')

            print(trans_data)
            logging.info(trans_data)
            print('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('달러 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
        else:
            print('달러 같은 매매구분 ... 거래 없음')
            logging.info('달러 같은 매매구분 ... 거래 없음')

    def send_result_to_email(self, msg):
        import smtplib
        from email.mime.text import MIMEText

        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.ehlo()  # say Hello
        smtp.starttls()  # TLS 사용시 필요
        smtp.login('bgchoi666@gmail.com', 'zxwzswqgfdvthsgu')

        msg = MIMEText(msg)
        msg['Subject'] = '달러, 주가 예측'
        msg['To'] = 'bgchoi666@gmail.com'
        smtp.sendmail('bgchoi666@gmail.com', 'bgchoi666@gmail.com', msg.as_string())

        smtp.quit()

    def kakao_update_trade(self, code, path):

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # ========== 데이터 파일 update ===============================
        df = pd.read_csv(path, encoding='euc-kr')

        if df['date'].values[-1] >= today:
            print("카카오 거래 종료")
            logging.info("카카오 거래 종료")
            return

        # 가격 데이터 가져오기
        self.kiwoom.reset_stock()
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "5001")
        time.sleep(0.2)

        cnt = 0
        while not self.kiwoom.stock_close:
            if cnt > 10:
                print("카카오 가격 정보 다운로드 실패")
                logging.info("카카오 가격 정보 다운로드 실패")
                return
            self.kiwoom.set_input_value("종목코드", code)
            self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "5001")
            time.sleep(0.2)
            cnt += 1

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
            FastK[i] = (df['종가'].values[len(df)-i-1] - min(df['저가'].values[-i-12:len(df)-i]))/(max(df['고가'].values[-i-12:(len(df)-i)]) - min(df['저가'].values[-i-12:len(df)-i])) * 100
        for i in range(5):
            FastD[i] = np.mean(FastK[i:i+5])
        df.loc[len(df)-1, 'SlowK'] = np.mean(FastD[:5])
        df.loc[len(df)-1, 'SlowD'] = np.mean(df['SlowK'].values[-5:])

        df.to_csv(path, index=False, encoding='euc-kr')


        # ==================== 앙상블 예측 ================================
        os.system("start cmd /c \"kakao_ensemble_predict.bat\"")
        time.sleep(0.5)

        now = datetime.datetime.now()

        while True:
            try:
                kakao_results_df = pd.read_csv("../stock_1D/models/kakao_1D_results.csv", encoding='euc-kr')
            except:
                print('pred_results.csv file read error...')
                if (datetime.datetime.now() - now).seconds > 60 * 5:
                    print('카카오 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                    logging.error('카카오 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                    return
                continue
            if kakao_results_df.loc[kakao_results_df.index.max(), 'dates'] == today:
                break
            if (datetime.datetime.now() - now).seconds > 60 * 5:
                print('카카오 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                logging.error('카카오 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                return

        # ==================== 거래 =======================================
        buy_sell_signal = int(kakao_results_df['results'].values[-1])
        if buy_sell_signal == 2:
            gubun = "매수"
        elif buy_sell_signal == 1:
            gubun = "매도"
        else:
            gubun = "중립"

        self.msg += ", 카카오: " + gubun

        # 주문 내역 거래내역 파일 (dollar_trans.csv) 에 저장
        trans_df = pd.read_csv('kakao_trans.csv', encoding='euc-kr')

        if int(trans_df['거래가격'].values[-1]) > 0 and int(trans_df['청산가격'].values[-1]) == 0:
            state = '매수'
        else:
            state = '보유없음'

        if state == '매수' and gubun == '매도':

            # 매도
            trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산일시'] = today
            trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산가격'] = str(self.kiwoom.stock_close)

            trans_df.to_csv('kakao_trans.csv', index=False, encoding='euc-kr')

            print('카카오 매도 ' + str(self.kiwoom.stock_close) + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('카카오 매도 ' + str(self.kiwoom.stock_close) + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')

        elif state == '보유없음' and gubun == '매수':
            # 매수
            trans_data = {'거래일시': today, '매매구분': gubun, '수량': 1,
                          '거래가격': self.kiwoom.stock_close, '청산일시': '0', '청산가격': '0'}
            trans_df = trans_df.append(trans_data, ignore_index=True)
            trans_df.to_csv('kakao_trans.csv', index=False, encoding='euc-kr')

            print(trans_data)
            logging.info(trans_data)
            print('카카오 신규 매수 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('카카오 신규 매수 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')

        # 매수 또는 보유 없음 유지
        else:
            print('카카오 ... 거래 없음')
            logging.info('카카오 ... 거래 없음')

    def samsung_update_trade(self, code, path):

        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # ========== 데이터 파일 update ===============================
        df = pd.read_csv(path, encoding='euc-kr')

        if df['date'].values[-1] >= today:
            print("삼성 거래 종료")
            logging.info("삼성 거래 종료")
            return

        # 가격 데이터 가져오기
        self.kiwoom.reset_stock()
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "5002")
        time.sleep(0.2)

        cnt = 0
        while not self.kiwoom.stock_close:
            if cnt > 10:
                print("삼성전자 가격 정보 다운로드 실패")
                logging.info("삼성전자 가격 정보 다운로드 실패")
                return
            self.kiwoom.set_input_value("종목코드", code)
            self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "5002")
            time.sleep(0.2)
            cnt += 1

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
            FastK[i] = (df['종가'].values[len(df)-i-1] - min(df['저가'].values[-i-12:len(df)-i]))/(max(df['고가'].values[-i-12:(len(df)-i)]) - min(df['저가'].values[-i-12:len(df)-i])) * 100
        for i in range(5):
            FastD[i] = np.mean(FastK[i:i+5])
        df.loc[len(df)-1, 'SlowK'] = np.mean(FastD[:5])
        df.loc[len(df)-1, 'SlowD'] = np.mean(df['SlowK'].values[-5:])

        df.to_csv(path, index=False, encoding='euc-kr')


        # ==================== 앙상블 예측 ================================
        os.system("start cmd /c \"samsung_ensemble_predict.bat\"")
        time.sleep(0.5)

        now = datetime.datetime.now()

        while True:
            try:
                samsung_results_df = pd.read_csv("../stock_1D/models/samsung_1D_results.csv", encoding='euc-kr')
            except:
                print('pred_results.csv file read error...')
                if (datetime.datetime.now() - now).seconds > 60 * 5:
                    print('삼성전자 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                    logging.error('삼성전자 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                    return
                continue
            if samsung_results_df.loc[samsung_results_df.index.max(), 'dates'] == today:
                break
            if (datetime.datetime.now() - now).seconds > 60 * 5:
                print('삼성전자 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                logging.error('삼성전자 예측 프로세스 실패... 시간 경과 5분... 프로그램 종료')
                return

        # ==================== 거래 =======================================
        buy_sell_signal = int(samsung_results_df['results'].values[-1])
        if buy_sell_signal == 2:
            gubun = "매수"
        elif buy_sell_signal == 1:
            gubun = "매도"
        else:
            gubun = "중립"

        self.msg += ", 삼성: " + gubun

        # 주문 내역 거래내역 파일 (dollar_trans.csv) 에 저장
        trans_df = pd.read_csv('samsung_trans.csv', encoding='euc-kr')

        if int(trans_df['거래가격'].values[-1]) > 0 and int(trans_df['청산가격'].values[-1]) == 0:
            state = '매수'
        else:
            state = '보유없음'

        if state == '매수' and gubun == '매도':

            # 매도
            trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산일시'] = today
            trans_df.loc[trans_df.loc[trans_df["매매구분"] == "매수"].index.max(), '청산가격'] = str(self.kiwoom.stock_close)

            trans_df.to_csv('samsung_trans.csv', index=False, encoding='euc-kr')

            print('삼성전자 매도 ' + str(self.kiwoom.stock_close) + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('삼성전자 매도 ' + str(self.kiwoom.stock_close) + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')

        elif state == '보유없음' and gubun == '매수':
            # 매수
            trans_data = {'거래일시': today, '매매구분': gubun, '수량': 1,
                          '거래가격': self.kiwoom.stock_close, '청산일시': '0', '청산가격': '0'}
            trans_df = trans_df.append(trans_data, ignore_index=True)
            trans_df.to_csv('samsung_trans.csv', index=False, encoding='euc-kr')

            print(trans_data)
            logging.info(trans_data)
            print('삼성전자 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')
            logging.info('삼성전자 신규 ' + gubun + ' 주문 완료... dollar_trans.csv 파일에 주문 내역 저장...')

        # 매수 or 보유 없음 유지
        else:
            print('삼성전자 ... 거래 없음')
            logging.info('삼성전자 ... 거래 없음')

    def profit_graph(self):
        if len(self.kiwoom.opw20007_output['multi']) > 0:
            last_purchase_price = self.kiwoom.opw20007_output['multi'][0][5]
        else:
            last_purchase_price = '0'
        os.system("start cmd /c \"C:\\Users\\user\\Anaconda3\\envs\\test\\python profit_graph.py " + self.lineEdit_4.text() + " " + last_purchase_price + "\" ")

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
        elif rqname == "일가격거래량_req":
            self._OPT50030(rqname, trcode)
        elif rqname == "미결제약정_req":
            self._OPT50062(rqname, trcode)
        elif rqname == 'OPW20002_req':
            self._OPW20002(rqname, trcode)
        elif rqname == 'OPW20009_req':
            self._OPW20009(rqname, trcode)
        elif rqname == 'OPW20005_req':
            self._OPW20005(rqname, trcode)
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

    def _opt50001(self, rqname, trcode):
        self.current_price = float((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))


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


    def _OPW20002(self, rqname, trcode):
        self.fee = int(self._comm_get_data(trcode, "", rqname, 0, "선물수수료"))
        self.sell_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물매도수량"))
        self.sell_price = int(self._comm_get_data(trcode, "", rqname, 0, "선물매도평균가격").lstrip('-'))*0.01
        self.buy_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물매수수량"))
        self.buy_price = int(self._comm_get_data(trcode, "", rqname, 0, "선물매수평균가격").lstrip('-'))*0.01
        self.resale_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물전매수량"))
        self.redemption_volumn = int(self._comm_get_data(trcode, "", rqname, 0, "선물환매수량"))

    def _OPW20005(self, rqname, trcode):
        self.출력건수 = self._comm_get_data(trcode, "", rqname, 0, "출력건수")
        self.매매구분 = self._comm_get_data(trcode, "", rqname, 0, "매매구분")
        self.주문수량 = int(self._comm_get_data(trcode, "", rqname, 0, "주문수량"))
        self.체결수량 = int(self._comm_get_data(trcode, "", rqname, 0, "체결수량"))
        self.미체결수량 = int(self._comm_get_data(trcode, "", rqname, 0, "미체결수량"))
        self.체결가 = float(self._comm_get_data(trcode, "", rqname, 0, "체결가"))/100
        self.약정시간 = self._comm_get_data(trcode, "", rqname, 0, "약정시간")

    def _OPW20007(self, rqname, trcode):
        self.reset_opw20007_output()

        # single data
        total_purchase_price = self._comm_get_data(trcode, "", rqname, 0, "약정금액합계")
        total_eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, 0, "평가손익금액")
        total_cnt = self._comm_get_data(trcode, "", rqname, 0, "출력건수")

        self.opw20007_output['single'].append(Kiwoom.change_format(total_purchase_price))
        self.opw20007_output['single'].append(Kiwoom.change_format(total_eval_profit_loss_price))
        self.opw20007_output['single'].append(Kiwoom.change_format(total_cnt))

        if self.opw20007_output['single'][0] == '0':
            return

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
            purchase_price = str(float(purchase_price) / 1000)
            current_price = str(float(current_price) / 1000)
            eval_profit_loss_price = Kiwoom.change_format(eval_profit_loss_price)
            earning_rate = str(earning_rate)

            self.opw20007_output['multi'].append([code, name, gubun, quantity, purchase_price, current_price,
                                                  eval_profit_loss_price, earning_rate])


    def _OPW20009(self, rqname, trcode):
        self.new_quantity = int(self._comm_get_data(trcode, "", rqname, 0, "신규가능수량"))
        self.liquid_quantity = int(self._comm_get_data(trcode, "", rqname, 0, "청산가능수량"))
        self.total_quantity = int(self._comm_get_data(trcode, "", rqname, 0, "총가능수량"))

    def _OPW20010(self, rqname, trcode):
        self.d2_deposit = self._comm_get_data(trcode, "", rqname, 0, "주문가능총액")
        self.margin = self._comm_get_data(trcode, "", rqname, 0, "증거금총액")


    def _OPT50029(self, rqname, trcode):
        self.reset_opt50029_output()

        # 시가, 고가, 저가, 종가, 거래량
        self.close = (self._comm_get_data(trcode, "", rqname, 1, "현재가")).lstrip('+-')
        self.open = (self._comm_get_data(trcode, "", rqname, 1, "시가")).lstrip('+-')
        self.high = (self._comm_get_data(trcode, "", rqname, 1, "고가")).lstrip('+-')
        self.low = (self._comm_get_data(trcode, "", rqname, 1, "저가")).lstrip('+-')
        self.volumn = self._comm_get_data(trcode, "", rqname, 1, "거래량").replace(',', '')

        #t = str(self._comm_get_data(trcode, "", rqname, 1, "체결시간"))
        #self.che_time = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

        self.che_time = (datetime.datetime.strptime(self._comm_get_data(trcode, "", rqname, 1, "체결시간"),
                                                    "%Y%m%d%H%M%S") + datetime.timedelta(hours=1)).strftime("%Y/%m/%d/%H:00")

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

            #t = str(self._comm_get_data(trcode, "", rqname, i, "체결시간"))
            #t = t[:4] + "/" + t[4:6] + "/" + t[6:8] + "/" + t[8:10] + ":00"

            t = (datetime.datetime.strptime(self._comm_get_data(trcode, "", rqname, i, "체결시간"),
                                                    "%Y%m%d%H%M%S") + datetime.timedelta(hours=1)).strftime("%Y/%m/%d/%H:00")

            self.opt50029_output['multi'].append([t, open, high, low, close, volumn])

    def _OPT50030(self, rqname, trcode):

        # 시가, 고가, 저가, 종가, 거래량
        self.close_1D = float((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))
        self.open_1D = float((self._comm_get_data(trcode, "", rqname, 0, "시가")).lstrip('+-'))
        self.high_1D = float((self._comm_get_data(trcode, "", rqname, 0, "고가")).lstrip('+-'))
        self.low_1D = float((self._comm_get_data(trcode, "", rqname, 0, "저가")).lstrip('+-'))
        self.volume_1D = int(self._comm_get_data(trcode, "", rqname, 0, "누적거래량"))

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

    def _OPT50019(self, rqname, trcode):

        try:
            # 시가, 고가, 저가, 종가
            self.dollar_open = float((self._comm_get_data(trcode, "", rqname, 0, "시가")).lstrip('+-'))
            self.dollar_high = float((self._comm_get_data(trcode, "", rqname, 0, "고가")).lstrip('+-'))
            self.dollar_low = float((self._comm_get_data(trcode, "", rqname, 0, "저가")).lstrip('+-'))
            self.dollar_close = float((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))
        except:
            print("달러 다운로드 실패")

    def _opt10001(self, rqname, trcode):
        try:
            self.item_name = self._comm_get_data(trcode, "", rqname, 0, "종목명")

            # 시가, 고가, 저가, 종가
            self.stock_open = int((self._comm_get_data(trcode, "", rqname, 0, "시가")).lstrip('+-'))
            self.stock_high = int((self._comm_get_data(trcode, "", rqname, 0, "고가")).lstrip('+-'))
            self.stock_low = int((self._comm_get_data(trcode, "", rqname, 0, "저가")).lstrip('+-'))
            self.stock_close = int((self._comm_get_data(trcode, "", rqname, 0, "현재가")).lstrip('+-'))
            self.stock_volume = int((self._comm_get_data(trcode, "", rqname, 0, "거래량")).lstrip('+-'))
        except:
            print("주식 다운로드 실패")

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

    def reset_opt50062_output(self):
        self.opt50062_output = {'multi': []}

    def reset_opt50029_output(self):
        self.opt50029_output = {'multi': []}

    def reset_opw20007_output(self):
        self.opw20007_output = {'single': [], 'multi': []}

    def reset_opt10080_output(self):
        self.opt10080_output = {'single': [], 'multi': []}

    def reset_sell_output(self):
        self.sell_output = {'single': [], 'multi': []}

    def reset_dollar(self):
        self.dollar_open = ''
        self.dollar_high = ''
        self.dollar_low = ''
        self.dollar_close = ''

    def reset_stock(self):
        self.stock_open = ''
        self.stock_high = ''
        self.stock_low = ''
        self.stock_close = ''
        self.stock_volume = ''
        self.item_name = ''


    def get_server_gubun(self):
        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow(sys.argv[1])
    myWindow.show()
    app.exec_()