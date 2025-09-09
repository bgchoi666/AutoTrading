# -*- coding:utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from Kiwoom import *
import PyGran
import datetime
import os

form_class = uic.loadUiType("pytrader.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self, gubun):
        super().__init__()

        # 0: 수동 주문  1: 자동 주문
        self.gubun = int(gubun)

        self.setupUi(self)

        self.trade_stocks_done = False

        self.kiwoom = Kiwoom()
        # login 화면 연결
        self.kiwoom.comm_connect()

        self.kiwoom.dynamicCall('GetConditionLoad()')  # 키움 서버에 사용자 조건식 목록을 요청

        self.lineEdit.textChanged.connect(self.code_changed)
        #self.lineEdit_3.textChanged.connect(self.code_changed_3)
        self.lineEdit_5.textChanged.connect(self.code_changed_5)
        self.lineEdit_6.textChanged.connect(self.code_changed_6)

        accouns_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
        accounts = self.kiwoom.get_login_info("ACCNO")

        accounts_list = accounts.split(';')[0:accouns_num]
        self.comboBox.addItems(accounts_list)

        self.tableWidget_2.clicked.connect(self.select_item_1)
        self.tableWidget_3.clicked.connect(self.select_item_2)

        self.pushButton.clicked.connect(self.send_order)
        self.pushButton_2.clicked.connect(self.check_balance)
        self.pushButton_3.clicked.connect(self.auto_change)
        self.pushButton_4.clicked.connect(self.trade_stocks)
        self.pushButton_5.clicked.connect(self.make_buy_list)
        self.pushButton_6.clicked.connect(self.make_sell_list)
        self.pushButton_7.clicked.connect(self.load_buy_sell_list)

        self.loss_cut = self.lineEdit_5.text()
        self.profit_real = self.lineEdit_6.text()

        # Timer1
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)

        # Timer2
        self.timer2 = QTimer(self)
        self.timer2.start(1000)
        self.timer2.timeout.connect(self.timeout2)

        self.load_buy_sell_list()

    def timeout(self):
        if self.gubun  == 1:
            self.pushButton_3.setText("자동 거래")
        else:
            self.pushButton_3.setText("수동 거래")

        current_time = QTime.currentTime()

        text_time = current_time.toString("hh:mm:ss")
        time_msg = "현재시간: " + text_time

        state = self.kiwoom.get_connect_state()
        if state == 1:
            state_msg = "서버 연결 중"
        else:
            state_msg = "서버 미 연결 중"

        self.statusbar.showMessage(state_msg + " | " + time_msg)

    def timeout2(self):
        current_time = QTime.currentTime()
        market_start_time = QTime(9, 0, 0)
        market_end_time = QTime(15, 15, 0)

        if self.checkBox.isChecked():
            self.check_balance()

        if self.gubun == 1 and current_time > market_start_time and current_time < market_end_time:
                if self.trade_stocks_done is False:
                    self.trade_stocks()
                    self.trade_stocks_done = True

                    # 매수 완료 종목을 sell_list에
                    f = open("buy_list.txt", 'rt', encoding='utf-8')
                    buy_list = f.readlines()
                    f.close()

                    f = open("sell_list.txt", 'wt', encoding='utf-8')
                    for row_data in buy_list:
                        row_data = row_data.replace("주문완료", "매도전")
                        f.writelines(row_data)
                    f.close()
                else:
                    self.trade_stocks3()

    def auto_change(self):
        if self.gubun == 1:
            self.pushButton_3.setText("수동 거래")
            self.gubun = 0
        else:
            self.pushButton_3.setText("자동 거래")
            self.gubun = 1

    def send_order(self):
        order_type_lookup = {'신규매수': 1, '신규매도': 2, '매수취소': 3, '매도취소': 4}
        hoga_lookup = {'지정가': "00", '시장가': "03"}

        account = self.comboBox.currentText()
        order_type = self.comboBox_2.currentText()
        code = self.lineEdit.text()
        hoga = self.comboBox_3.currentText()
        num = self.spinBox.value()
        price = self.spinBox_2.value()

        self.kiwoom.send_order("send_order_req", "0101", account, order_type_lookup[order_type], code, num, price,
                               hoga_lookup[hoga], "")

    def get_item_info(self, code):
        self.kiwoom.set_input_value('종목코드', code)
        self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
        time.sleep(0.2)

        year_high = self.kiwoom.opt10001_연중최고
        self.label_12.setText(year_high)

        year_low = self.kiwoom.opt10001_연중최저
        self.label_14.setText(year_low)

        # PER
        PER = self.kiwoom.opt10001_PER
        self.label_16.setText(PER)

        # PBR
        PBR = self.kiwoom.opt10001_PBR
        self.label_18.setText(PBR)

        # ROE
        ROE = self.kiwoom.opt10001_ROE
        self.label_8.setText(ROE)

        # 현재가
        self.spinBox_2.setValue(abs(int(self.kiwoom.opt10001_현재가)))

    def select_item_1(self):
        idx = self.tableWidget_2.currentRow()
        self.lineEdit.setText(self.kiwoom.opw00018_output['multi'][idx][6][1:])

        # 공통종목 정리
        self.commonCodes = []
        for i in range(len(self.kiwoom.condCodes[2])):
            if self.kiwoom.condCodes[2][i] in self.kiwoom.condCodes[1] and self.kiwoom.condCodes[2][i] in self.kiwoom.condCodes[3]:
                self.commonCodes.append(self.kiwoom.condCodes[2][i])
        print("조건식 검색 완료")

        self.get_item_info(self.kiwoom.opw00018_output['multi'][idx][6][1:])

    def select_item_2(self):
        i = self.tableWidget_3.currentRow()

        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.readlines()
        f.close()

        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.readlines()
        f.close()

        buy_count = len(buy_list)

        if i < buy_count:
            code = buy_list[i].split(';')[1].rstrip()
            self.lineEdit.setText(code)
        else:
            code = sell_list[i-buy_count].split(';')[1].rstrip()
            self.lineEdit.setText(code)

        self.get_item_info(code)

    def code_changed(self):
        code = self.lineEdit.text()
        name = self.kiwoom.get_master_code_name(code)
        self.lineEdit_2.setText(name)

    #def code_changed_3(self):
    #    code = self.lineEdit_3.text()
    #    name = self.kiwoom.get_master_code_name(code)
    #    self.lineEdit_4.setText(name)

    def code_changed_5(self):
        self.loss_cut = self.lineEdit_5.text()
        self.profit_real = self.lineEdit_6.text()
        self.label_9.setText("매도 종목 : 수익률 " + str(self.loss_cut) + "% 이하 손절, " + str(self.profit_real) + "% 이상 이익 실현")

    def code_changed_6(self):
        self.profit_real = self.lineEdit_6.text()
        self.loss_cut = self.lineEdit_5.text()
        self.label_9.setText("매도 종목 : 수익률 " + str(self.loss_cut) + "% 이하 손절, " + str(self.profit_real) + "% 이상 이익 실현")

    def check_balance(self):
        self.kiwoom.reset_opw00018_output()
        account_list = self.kiwoom.get_login_info("ACCNO").split(';')
        for i in range(len(account_list)):
            if account_list[i][8:] == '10' or '11':
                account_number = account_list[i]
                break

        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.set_input_value("조회구분", 2)
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account_number)
            self.kiwoom.set_input_value("조회구분", 2)
            self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 2, "2000")

        # opw00001
        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("opw00001_req", "opw00001", 0, "1000")

        # balance
        item = QTableWidgetItem(self.kiwoom.change_format(self.kiwoom.d2_deposit))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 0, item)

        for i in range(1, 6):
            item = QTableWidgetItem(self.kiwoom.opw00018_output['single'][i - 1])
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.tableWidget.setItem(0, i, item)

        self.tableWidget.resizeRowsToContents()

        # Item list
        item_count = len(self.kiwoom.opw00018_output['multi'])
        self.tableWidget_2.setRowCount(item_count)

        for j in range(item_count):
            row = self.kiwoom.opw00018_output['multi'][j]
            for i in range(len(row)):
                item = QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.tableWidget_2.setItem(j, i, item)

        self.tableWidget_2.resizeRowsToContents()

        # Item list 일부 종목만 표시
        #item_count = len(self.kiwoom.opw00018_output['multi'])
        #self.tableWidget_2.setRowCount(5)

        #n = 0
        #for j in range(item_count):
        #    row = self.kiwoom.opw00018_output['multi'][j]
        #    if row[0] not in ['현대차', 'POSCO홀딩스', '삼성전자', 'LG전자', '넥스트칩']:
        #        continue
        #    for i in range(len(row)):
        #        item = QTableWidgetItem(row[i])
        #        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        #        self.tableWidget_2.setItem(n, i, item)
        #    n += 1

        #self.tableWidget_2.resizeRowsToContents()

    def granville_buy_signal(self):
        code = self.lineEdit_3.text()
        today = datetime.datetime.today().strftime("%Y%m%d")

        df = PyGran.get_ohlcv
        time.sleep(0.2)
        if PyGran.granville_signal(code, df):
            self.label_8.setText("Yes!!!!!!")
        else:
            self.label_8.setText("No!!!!!!")

    def make_buy_list(self):
        f = open("buy_list.txt", "wt", encoding="utf-8")
        f.close()
        os.system("start cmd /k \"F:\\알고리즘트레이딩\\PyGran\"")
        self.label_9.setText("\ngranville 매수 종목 search 시작.....  종료후 프로그램 restart  하세요....  ")

    def make_sell_list(self):
        p = "\"" + "C:\\anaconda3_32\\python C:\\알고리즘트레이딩\\PySell.py " + self.loss_cut + "  " + self.profit_real + "\""
        os.system("start cmd /k " + p)
        self.label_9.setText("\nsell list 작성 중......\n종료후 program restart 하세요......")

    def load_buy_sell_list(self):
        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.readlines()
        f.close()

        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.readlines()
        f.close()

        row_count = len(buy_list) + len(sell_list)
        self.tableWidget_3.setRowCount(row_count)

        # buy list
        for j in range(len(buy_list)):
            row_data = buy_list[j]
            split_row_data = row_data.split(';')
            split_row_data[1] = self.kiwoom.get_master_code_name(split_row_data[1].rsplit())

            for i in range(len(split_row_data)):
                item = QTableWidgetItem(split_row_data[i].rstrip())
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
                self.tableWidget_3.setItem(j, i, item)

        # sell list
        for j in range(len(sell_list)):
            row_data = sell_list[j]
            split_row_data = row_data.split(';')
            split_row_data[1] = self.kiwoom.get_master_code_name(split_row_data[1].rstrip())

            for i in range(len(split_row_data)):
                item = QTableWidgetItem(split_row_data[i].rstrip())
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
                self.tableWidget_3.setItem(len(buy_list) + j, i, item)

        self.tableWidget_3.resizeRowsToContents()

    def trade_stocks(self):
        self.kiwoom.reset_opw00018_output()
        account_list = self.kiwoom.get_login_info("ACCNO")
        account = account_list.split(';')[0]
        if account[8:] != '10' and account[8:] != '11':
            account = account_list.split(';')[1]

        self.kiwoom.set_input_value("계좌번호", account)
        self.kiwoom.set_input_value("조회구분", 2)
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")
        time.sleep(0.2)

        hoga_lookup = {'지정가': "00", '시장가': "03"}

        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.readlines()
        f.close()

        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.readlines()
        f.close()

        # account
        account = self.comboBox.currentText()

        # sell order
        for i, row_data in enumerate(sell_list):
            split_row_data = row_data.split(';')
            hoga = split_row_data[2]
            code = split_row_data[1]
            find = False
            for item in self.kiwoom.opw00018_output['multi']:
                if code != item[6][1:]:
                    continue
                num = item[1]
                find = True
                break
            if not find:
                continue
            price = split_row_data[4]
            state = split_row_data[5]

            if state.rstrip() == '매도전':
                self.kiwoom.send_order("send_order_req", "0101", account, 2, code, num, price,
                                       hoga_lookup[hoga], "")
                sell_list[i] = sell_list[i].replace("매도전", "주문완료")

        buyCount = len(buy_list)

        # buy order
        for i, row_data in enumerate(buy_list):
            # d+2 추정 예수금 십만원 이하이면 매수 중단
            self.kiwoom.set_input_value("계좌번호", account)
            self.kiwoom.comm_rq_data("opw00001_req", "opw00001", 0, "1000")
            time.sleep(0.2)

            d2_deposit = int(self.kiwoom.d2_deposit)
            if float(d2_deposit) < 1000000:
                break

            max_amt_per_item = int(d2_deposit / buyCount)

            split_row_data = row_data.split(';')
            hoga = split_row_data[2]
            code = split_row_data[1]

            # 현재가 조회 십만원치 수량 계산
            self.kiwoom.set_input_value('종목코드', code)
            self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
            time.sleep(0.2)
            if code != self.kiwoom.opt10001_종목코드:
                continue
            else:
                num = int(max_amt_per_item / abs(int(self.kiwoom.opt10001_현재가)))

            if num < 1: continue

            price = split_row_data[4]
            state = split_row_data[5]

            if state.rstrip() == '매수전':
                self.kiwoom.send_order("send_order_req", "0101", account, 1, code, num, price,
                                       hoga_lookup[hoga], "")

                buy_list[i] = buy_list[i].replace("매수전", "주문완료")

        # buy_list ile update
        f = open("buy_list.txt", 'wt', encoding='utf-8')
        for row_data in buy_list:
            f.writelines(row_data)
        f.close()

        # sell_list file update
        f = open("sell_list.txt", 'wt', encoding='utf-8')
        for row_data in sell_list:
            f.writelines(row_data)
        f.close()

    def trade_stocks2(self):
        self.kiwoom.reset_opw00018_output()
        account_list = self.kiwoom.get_login_info("ACCNO")
        account = account_list.split(';')[1]
        if account[8:] == '31':
            account = account_list.split(';')[0]

        self.kiwoom.set_input_value("계좌번호", account)
        self.kiwoom.set_input_value("조회구분", 2)
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")
        time.sleep(0.2)

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account)
            self.kiwoom.set_input_value("조회구분", 2)
            self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 2, "2000")

        # sell order
        for item in self.kiwoom.opw00018_output['multi']:
            code = item[6][1:]
            num = item[1]
            earning_rate = float(item[5])

            if earning_rate < float(self.loss_cut) or earning_rate > float(self.profit_real):
                self.kiwoom.send_order("send_order_req", "0101", account, 2, code, num, 0, "03", "")

    def trade_stocks3(self):
        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.readlines()
        f.close()

        self.kiwoom.reset_opw00018_output()
        account_list = self.kiwoom.get_login_info("ACCNO")
        account = account_list.split(';')[1]
        if account[8:] == '31':
            account = account_list.split(';')[0]

        self.kiwoom.set_input_value("계좌번호", account)
        self.kiwoom.set_input_value("조회구분", 2)
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")
        time.sleep(0.2)

        # sell order
        for i, row_data in enumerate(sell_list):
            split_row_data = row_data.split(';')
            code = split_row_data[1]
            find = False
            for item in self.kiwoom.opw00018_output['multi']:
                if code != item[6][1:]:
                    continue
                find = True
                break
            if not find:
                continue

            earning_rate = float(item[5])
            code = item[6][1:]
            num = item[1]
            state = split_row_data[5]

            if state.rstrip() != '매도전':
                continue

            current_time = QTime.currentTime()
            market_end_time = QTime(11, 0, 0)

            if current_time < market_end_time:
                if earning_rate < float(self.loss_cut) or earning_rate > float(self.profit_real):
                    self.kiwoom.send_order("send_order_req", "0101", account, 2, code, num, 0, "03", "")
                    sell_list[i] = sell_list[i].replace("매도전", "주문완료")
            else:
                self.kiwoom.send_order("send_order_req", "0101", account, 2, code, num, 0, "03", "")
                sell_list[i] = sell_list[i].replace("매도전", "주문완료")

        # sell_list file update
        f = open("sell_list.txt", 'wt', encoding='utf-8')
        for row_data in sell_list:
            f.writelines(row_data)
        f.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow(sys.argv[1])
    myWindow.show()
    app.exec_()