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
import pandas_datareader.data as web
from datetime import date

form_class = uic.loadUiType("futureTrader.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self, gubun):
        super().__init__()

        self.setupUi(self)

        self.kiwoom = Kiwoom()
        # login 화면 연결
        self.kiwoom.comm_connect()

        #2일전 ~ 전일sp500 지수를 가져온다
        today = datetime.datetime.today().strftime("%m/%d/%Y")
        three_days_before = (datetime.datetime.today() + timedelta(days=-3)).strftime("%m/%d/%Y")
        self.sp500 = web.DataReader('^GSPC', data_source='yahoo', start=three_days_before, end=today)

        accouns_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
        accounts = self.kiwoom.get_login_info("ACCNO")

        accounts_list = accounts.split(';')[0:accouns_num]
        self.acc_no = accounts_list[1]
        self.comboBox.addItems(accounts_list)

        self.loss_cut = self.lineEdit.text()
        self.profit_real = self.lineEdit_3.text()


        today = date.today()
        #d = today.strftime("%Y-%m-%d")
        d = '2020-12-22'

        df = pd.read_csv("kospi200f_predict.csv")
        self.updown = int(df.loc[df['date'] == d]['updown'].values[0])

        # 0: 수동 주문  1: 자동 주문
        self.auto_gubun = 0
        if int(gubun) == 1:
            self.start_auto()
            self.new_order()

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
        self.timer2.start(1000 * 10)
        self.timer2.timeout.connect(self.timeout2)

    def timeout(self):

        current_time = QTime.currentTime()

        trading_state = ""
        if self.auto_gubun == 1:
            self.auto_liquidation()
            trading_state = "자동 청산 실행 중"
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
        if self.checkBox.isChecked():
            self.check_balance()

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
            row = self.kiwoom.opw20007_output['multi'][j][2:]
            for i in range(len(row)):
                item = QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.tableWidget_2.setItem(j, i, item)

        self.tableWidget_2.resizeRowsToContents()

    def stop_auto(self):
        self.auto_gubun = 0

    def start_auto(self):
        market_start_time = QTime(9, 0, 0)
        market_end_time = QTime(15, 45, 0)
        current_time = QTime.currentTime()

        if market_start_time > current_time:
            print('시장 개시 전....')
            return
        elif market_end_time < current_time:
            print('시장 종료....')
            return

        self.auto_gubun = 1

    def auto_liquidation(self):
        market_end_time = QTime(15, 29, 0)
        current_time = QTime.currentTime()

        # 잔고 조회후 청산 process
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        # 손절, 이익실현 값에 따른 청산, 15:30 이후 무조건 청산
        for row in self.kiwoom.opw20007_output['multi']:
            # 매수 청산
            if row[2] == "2":
                if (float(row[5]) - float(row[4])) / float(row[4]) * 100 > float(self.profit_real) or (float(row[5]) - float(row[4])) / float(row[4]) * 100 < float(self.loss_cut)  \
                or current_time > market_end_time:
                    gubun = "1"
                    self.kiwoom.dynamicCall("SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                                 ["send_order_req", "0101", self.acc_no, row[0], 1, gubun, "3", int(row[3]), "", ""])
            # 매도 청산
            else:
                if (float(row[4]) - float(row[5])) / float(row[4]) * 100 > float(self.profit_real) or (float(row[4]) - float(row[5])) / float(row[4]) * 100 < float(self.loss_cut)  \
                or current_time > market_end_time:
                    gubun = "2"
                    self.kiwoom.dynamicCall("SendOrderFO(QString, QString, QString, QString, int, QString, QString, int, QString, QString)",
                                 ["send_order_req", "0101", self.acc_no, row[0], 1, gubun, "3", int(row[3]), "", ""])
        # 잔고 조회후 보유 수량 0이면 신규거래
        #self.kiwoom.set_input_value("계좌번호", self.acc_no)
        #self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        #time.sleep(0.2)


    def new_order(self):
        self.first_order = False

        # 잔고 조회
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        # 주문가능총액
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20010_req", "OPW20010", 0, "1000")
        time.sleep(0.2)

        # 주문 가능 총액이 천만원 미만이면  진입 불가
        amt = float(self.kiwoom.d2_deposit)
        if amt < 10000000:
            return

        if self.updown == -1:
            # sell order
            hoga = "시장가"
            code = "101QC000"
            num = 1
            price = ""
            self.kiwoom.send_order("send_order_req", "0101", self.acc_no, code, 1, "1", "3", num, price, "")
        elif self.updown == 1:
            # buy order
            hoga = "시장가"
            code = "101QC000"
            num = 1
            price = ""
            self.kiwoom.send_order("send_order_req", "0101", self.acc_no, code, 1, "2", "3", num, price, "")

        time.sleep(0.2)

        # 체결후 잔고 조회
        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20007_req", "OPW20007", 0, "2000")
        time.sleep(0.2)

        self.kiwoom.set_input_value("계좌번호", self.acc_no)
        self.kiwoom.comm_rq_data("OPW20010_req", "OPW20010", 0, "1000")
        time.sleep(0.2)

TR_REQ_TIME_INTERVAL = 0.2

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

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

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

            self.opw20007_output['multi'].append([code, gubun, name, quantity, purchase_price, current_price,
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
        strip_data = data.lstrip('-0')
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