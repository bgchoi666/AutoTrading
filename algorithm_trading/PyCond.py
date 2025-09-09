# -*- coding:utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from Kiwoom import *
import time
import pandas as pd
from pandas import DataFrame
import datetime
import os
import random
#from config.kiwoomType import *
#from config.log_class import *

form_class = uic.loadUiType("condition.ui")[0]

class condWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.kiwoom = Kiwoom()
        self.kiwoom.comm_connect()

        self.kiwoom.dynamicCall('GetConditionLoad()')  # 키움 서버에 사용자 조건식 목록을 요청
        time.sleep(0.2)

        self.pushButton.clicked.connect(self.select_cond_item)
        self.pushButton_2.clicked.connect(self.refresh)
        self.pushButton_3.clicked.connect(self.search_items)
        self.pushButton_4.clicked.connect(self.clear_selection)
        self.pushButton_5.clicked.connect(self.delete_item)
        self.pushButton_6.clicked.connect(self.delete_item2)
        self.pushButton_7.clicked.connect(self.add_buy_list)
        self.pushButton_8.clicked.connect(self.load_buy_list)
        self.pushButton_9.clicked.connect(self.reset_buy_list)
        self.pushButton_10.clicked.connect(self.add_sell_list)
        self.pushButton_11.clicked.connect(self.load_sell_list)
        self.pushButton_13.clicked.connect(self.reset_sell_list)

        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.clicked.connect(self.get_item_info)
        self.item_info_stat = 0

        self.conditionNames = ['외국인지분율상승', '외국인순매수', '매수', '가치주', '매도', '중소형저평가우량주',
                               '대형저평가우량주', '실적호전주', '성장주', '수익성이좋은기업', '시총대비영업이익30%이상',
                               '당일단타 검색기', '강한 단타용 검색기', '종가베팅 검색기', '전환선 눌림목', '급등주', '보유종목', 'buy_list', 'sell_list']

        self.first_refresh = 1

    def delete_item(self):
        row = self.listWidget_2.currentRow()
        self.listWidget_2.takeItem(row)

    def delete_item2(self):
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)

    def clear_selection(self):
        self.listWidget_2.clear()

    def get_item_info(self):
        row = self.tableWidget.currentRow()
        code = self.tableWidget.item(row, 0).text()

        self.kiwoom.set_input_value('종목코드', code)
        self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
        time.sleep(0.2)

        if self.item_info_stat == 0:
            price = self.kiwoom.opt10001_현재가
            item = QTableWidgetItem(price)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(row, 2, item)

            price = self.kiwoom.opt10001_연중최고
            item = QTableWidgetItem(price)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(row, 3, item)

            price = self.kiwoom.opt10001_연중최저
            item = QTableWidgetItem(price)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(row, 4, item)

            self.item_info_stat = 1
        else:
            # PER
            PER = self.kiwoom.opt10001_PER
            item = QTableWidgetItem(PER)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(row, 2, item)

            # PBR
            PBR = self.kiwoom.opt10001_PBR
            item = QTableWidgetItem(PBR)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(row, 3, item)

            # PBR + ROE
            ROE = self.kiwoom.opt10001_ROE
            item = QTableWidgetItem(ROE)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(row, 4, item)

            self.item_info_stat = 0

        self.tableWidget.resizeRowsToContents()

    def show_condition_name_list(self):
        #time.sleep(2)
        #if "보유종목" not in self.kiwoom.conditionNames:
        #    self.holding_items()
        for i in range(len(self.kiwoom.conditionNames)):
            self.listWidget.addItem(self.kiwoom.conditionNames[i])

        # 매수 매도 시나리오
        #self.listWidget_2.addItem('매도')
        #self.listWidget_2.addItem('보유종목')

        #self.search_items()
        #self.reset_sell_list()

        #self.clear_selection()
        #self.listWidget_2.addItem('매수')
        #self.listWidget_2.addItem('가치주')
        #self.listWidget_2.addItem('시총대비영업이익30%이상')
        #self.search_items()
        #self.reset_buy_list()

    def select_cond_item(self):
        item = self.listWidget.currentItem().text()
        self.listWidget_2.addItem(item)

    def refresh(self):
        self.listWidget.clear()

        # condintionCodes 에서 보유종목, buy, sell_list 제거
        if '보유종목' in self.kiwoom.conditionNames:
            del self.kiwoom.condCodes[-3:]

        self.holding_items()
        self.show_condition_name_list()

    def search_items(self):
        self.tableWidget.setRowCount(0)
        index_list = []
        for i in range(self.listWidget_2.count()):
            item = self.listWidget_2.item(i).text()
            index = self.kiwoom.conditionNames.index(item)
            #index = self.conditionNames.index(item)
            index_list.append(index)
        codes = self.get_codes(index_list)
        self.load_search_items(codes)

    def get_codes(self, index_list):
        # 공통종목 찾기
        codes = []
        for i in range(len(self.kiwoom.condCodes[index_list[0]])):
            isCommon = True
            for j in range(1, len(index_list)):
                if not self.kiwoom.condCodes[index_list[0]][i] in self.kiwoom.condCodes[index_list[j]]:
                    isCommon = False
                    break
            if isCommon:
                codes.append(self.kiwoom.condCodes[index_list[0]][i])
        return codes

    def init_buy_list(self):

        f = open("buy_list.txt", "wt", encoding='utf-8')

        df = pd.read_csv('buy_list.csv', encoding='euc-kr')
        for i in range(len(df)):
            code = str(df['종목코드'].values[i]).zfill(6)
            name = df['종목명'].values[i]
            f.writelines("매수;" + code + ";시장가;10;0;매수전;" + name + "\n")
        f.close()

    def add_buy_list(self):

        f = open("buy_list.txt", "at", encoding='utf-8')

        # 선택된 종목만 add
        row = self.tableWidget.currentRow()
        code = self.tableWidget.item(row, 0).text()
        name = self.tableWidget.item(row, 1).text()
        f.writelines("매수;" + code + ";시장가;10;0;매수전;" + name + "\n")

        # table의 전체 종목 add
        #rowCount = self.tableWidget.rowCount()
        #for i in range(rowCount):
        #    code = self.tableWidget.item(i, 0).text()
        #    name = self.tableWidget.item(i, 1).text()
        #    f.writelines("매수;" + code + ";시장가;10;0;매수전;" + name + "\n")

        f.close()

        self.sort_buy_list()

    def reset_buy_list(self):
        rowCount = self.tableWidget.rowCount()

        f = open("buy_list.txt", "wt", encoding='utf-8')

        for i in range(rowCount):
            code = self.tableWidget.item(i, 0).text()
            name = self.tableWidget.item(i, 1).text()
            f.writelines("매수;" + code + ";시장가;10;0;매수전;" + name + "\n")
        f.close()

        self.sort_buy_list()

    def add_sell_list(self):
        rowCount = self.tableWidget.rowCount()

        f = open("sell_list.txt", "at", encoding='utf-8')

        for i in range(rowCount):
            code = self.tableWidget.item(i, 0).text()
            name = self.tableWidget.item(i, 1).text()
            f.writelines("매도;" + code + ";시장가;10;0;매도전;" + name + "\n")
        f.close()

        self.sort_sell_list()

    def reset_sell_list(self):
        rowCount = self.tableWidget.rowCount()

        f = open("sell_list.txt", "wt", encoding='utf-8')

        for i in range(rowCount):
            code = self.tableWidget.item(i, 0).text()
            name = self.tableWidget.item(i, 1).text()
            f.writelines("매도;" + code + ";시장가;10;0;매도전;" + name + "\n")
        f.close()

        self.sort_sell_list()

    def load_buy_list(self):
        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.read().splitlines()
        f.close()

        self.tableWidget.setRowCount(len(buy_list))

        for i in range(len(buy_list)):
            code = buy_list[i].split(';')[1]

            # PER, PBR, ROE 정보 가져오기
            self.kiwoom.set_input_value('종목코드', code)
            self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
            time.sleep(0.2)

            # 종목코드
            code = self.kiwoom.opt10001_종목코드
            item = QTableWidgetItem(code)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 0, item)

            # 종목명
            name = self.kiwoom.opt10001_종목명
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 1, item)

            # PER
            PER = self.kiwoom.opt10001_PER
            item = QTableWidgetItem(PER)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 2, item)

            # PBR
            PBR = self.kiwoom.opt10001_PBR
            item = QTableWidgetItem(PBR)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 3, item)

            # PBR + ROE
            ROE = self.kiwoom.opt10001_ROE
            item = QTableWidgetItem(ROE)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 4, item)

        self.tableWidget.resizeRowsToContents()

    def load_sell_list(self):
        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.read().splitlines()
        f.close()

        self.tableWidget.setRowCount(len(sell_list))

        for i in range(len(sell_list)):
            code = sell_list[i].split(';')[1]

            # PER, PBR, ROE 정보 가져오기
            self.kiwoom.set_input_value('종목코드', code)
            self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
            time.sleep(0.2)

            # 종목코드
            code = self.kiwoom.opt10001_종목코드
            item = QTableWidgetItem(code)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 0, item)

            # 종목명
            name = self.kiwoom.opt10001_종목명
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 1, item)

            # PER
            PER = self.kiwoom.opt10001_PER
            item = QTableWidgetItem(PER)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 2, item)

            # PBR
            PBR = self.kiwoom.opt10001_PBR
            item = QTableWidgetItem(PBR)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 3, item)

            # PBR + ROE
            ROE = self.kiwoom.opt10001_ROE
            item = QTableWidgetItem(ROE)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(i, 4, item)

        self.tableWidget.resizeRowsToContents()

    def load_search_items(self, codes):
        self.tableWidget.setSortingEnabled(False)

        row_count = len(codes)
        self.tableWidget.setRowCount(row_count)

        # search item list
        for j, code in enumerate(codes):
            item = QTableWidgetItem(code)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(j, 0, item)

            name = self.kiwoom.get_master_code_name(code)
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(j, 1, item)

            # PER, PBR, ROE 정보 가져오기
            self.kiwoom.set_input_value('종목코드', code)
            self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
            time.sleep(0.2)

            # PER
            PER = self.kiwoom.opt10001_PER
            item = QTableWidgetItem(PER)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(j, 2, item)

            # PBR
            PBR = self.kiwoom.opt10001_PBR
            item = QTableWidgetItem(PBR)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(j, 3, item)

            # PBR + ROE
            ROE = self.kiwoom.opt10001_ROE
            item = QTableWidgetItem(ROE)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.tableWidget.setItem(j, 4, item)

        self.tableWidget.resizeRowsToContents()
        self.tableWidget.setSortingEnabled(True)

    def sort_buy_list(self):
        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.read().splitlines()
        f.close()

        data = []
        for row_data in buy_list:
            split_row_data = row_data.split(';')[:7]
            code = split_row_data[1]

            # PER, PBR, ROE 정보 추가
            self.kiwoom.set_input_value('종목코드', code)
            self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
            time.sleep(0.2)
            if self.kiwoom.opt10001_PER == "" or self.kiwoom.opt10001_ROE == "" or self.kiwoom.opt10001_PBR == "":
                eval = 999
            else:
                eval = float(self.kiwoom.opt10001_PER) - float(self.kiwoom.opt10001_ROE) + float(self.kiwoom.opt10001_PBR)
            split_row_data.append(eval)
            data.append(split_row_data)
        data.sort(key=lambda x: x[7])

        # save the ordered results
        f = open("buy_list.txt", "wt", encoding='utf-8')
        for i in range(len(data)):
            f.writelines("매수;" + data[i][1] + ";시장가;10;0;매수전;" + data[i][6] + ";" + str(data[i][7]) + "\n")
        f.close()

    def sort_sell_list(self):
        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.read().splitlines()
        f.close()

        data = []
        for row_data in sell_list:
            split_row_data = row_data.split(';')[:7]
            code = split_row_data[1]

            # PER, PBR, ROE 정보 추가
            self.kiwoom.set_input_value('종목코드', code)
            self.kiwoom.comm_rq_data('opt10001_req', 'opt10001', 0, 4000)
            time.sleep(0.2)
            if self.kiwoom.opt10001_PER == "" or self.kiwoom.opt10001_ROE == "" or self.kiwoom.opt10001_PBR == "":
                eval = 999
            else:
                eval = float(self.kiwoom.opt10001_PER) - float(self.kiwoom.opt10001_ROE) + float(self.kiwoom.opt10001_PBR)
            split_row_data.append(eval)
            data.append(split_row_data)
        data.sort(key=lambda x: x[7], reverse=True)

        # save the ordered results
        f = open("sell_list.txt", "wt", encoding='utf-8')
        for i in range(len(data)):
            f.writelines("매도;" + data[i][1] + ";시장가;10;0;매도전;" + data[i][6] + ";" + str(data[i][7]) + "\n")
        f.close()

    #def holding_items_from_file(self):

    def holding_items(self):

        if '보유종목' not in self.kiwoom.conditionNames:
            self.kiwoom.conditionNames.append("보유종목")
        self.kiwoom.reset_opw00018_output()
        account_numbers = self.kiwoom.get_login_info("ACCNO")
        account_number = account_numbers.split(';')[1]
        c = account_number[8:]
        if account_number[8:] != '10' and account_number[8:] != '11':
            account_number = account_numbers.split(';')[0]

        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.set_input_value("조회구분", 2)
        #self.kiwoom.set_input_value("거래소구분", 'KRX')
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account_number)
            self.kiwoom.set_input_value("조회구분", 2)
            self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 2, "2000")

        # Item list
        item_count = len(self.kiwoom.opw00018_output['multi'])

        codes = []
        for j in range(item_count):
            code = self.kiwoom.opw00018_output['multi'][j][6]
            codes.append(code[1:])
        self.kiwoom.condCodes.append(codes)

        """
        # 보유종목 list appended
        self.kiwoom.conditionNames.append("보유종목")
        f = open("보유list.txt", 'rt', encoding='utf-8')
        holding_list = f.read().splitlines()
        f.close()

        codes = []
        for i in range(len(holding_list)):
            codes.append(holding_list[i].split(';')[1])
        self.kiwoom.condCodes.append(codes)
        """

        # buy_list alppended
        if 'buy_list' not in self.kiwoom.conditionNames:
        #    self.init_buy_list()
            self.kiwoom.conditionNames.append("buy_list")
        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.read().splitlines()
        f.close()

        codes = []
        for i in range(len(buy_list)):
            codes.append(buy_list[i].split(';')[1])
        self.kiwoom.condCodes.append(codes)

        # sell_list appended
        if 'sell_list' not in self.kiwoom.conditionNames:
            self.kiwoom.conditionNames.append("sell_list")
        f = open("sell_list.txt", 'rt', encoding='utf-8')
        sell_list = f.read().splitlines()
        f.close()

        codes = []
        for i in range(len(sell_list)):
            codes.append(sell_list[i].split(';')[1])
        self.kiwoom.condCodes.append(codes)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    condWindow = condWindow()
    condWindow.show()
    condWindow.show_condition_name_list()
    app.exec_()
