import sys
from PyQt5.QtWidgets import *
import Kiwoom
import time
import pandas as pd
from pandas import DataFrame
import datetime
import random

MARKET_KOSPI   = 0
MARKET_KOSDAQ  = 10

class PySell:
    def __init__(self):
        self.kiwoom = Kiwoom.Kiwoom()
        self.kiwoom.comm_connect()
        time.sleep(0.2)

        state = self.kiwoom.get_connect_state()
        if state == 1:
            print("서버 연결 중")
        else:
            print("서버 미 연결 중")


    def check_sell(self, loss_cut, profit_real):
        sell_list =  []

        self.kiwoom.reset_sell_output()
        account_number = self.kiwoom.get_login_info("ACCNO")
        account_number = account_number.split(';')[0]

        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("sell_req", "opw00018", 0, "2000")

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account_number)
            self.kiwoom.comm_rq_data("sell_req", "opw00018", 2, "2000")

        # select sell items and put into sell_list
        item_count = len(self.kiwoom.sell_output['multi'])

        for j in range(item_count):
            row = self.kiwoom.sell_output['multi'][j]
            print(row)
            if float(row[1]) < float(loss_cut) or float(row[1]) > float(profit_real):
                sell_list.append([row[0], row[2]])

        self.update_sell_list(sell_list)


    def update_sell_list(self, sell_list):
        f = open("sell_list.txt", "wt", encoding='utf-8')
        for sell in sell_list:
            name = self.kiwoom.get_master_code_name(sell[0])
            f.writelines("매도;" + sell[0] + ";시장가;" + sell[1] + ";0;매도전;" + name + "\n")
        f.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pysell = PySell()
    pysell.check_sell(sys.argv[1], sys.argv[2])


