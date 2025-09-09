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

class PyMon:
    def __init__(self):
        self.kiwoom = Kiwoom.Kiwoom()
        self.kiwoom.comm_connect()

    def get_code_list(self, market):
        if market == 'kospi':
            codes = self.kiwoom.get_code_list_by_market(MARKET_KOSPI)
        elif market == 'kosdaq':
            codes = self.kiwoom.get_code_list_by_market(MARKET_KOSDAQ)
        elif market == 'kospi200':
            df = pd.read_excel('종목코드표.xlsx')
            codes = list(df['kospi200_code'])
        return codes

    def get_ohlcv(self, code, start):
        self.kiwoom.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("기준일자", start)
        self.kiwoom.set_input_value("수정주가구분", 1)
        self.kiwoom.comm_rq_data("opt10081_req", "opt10081", 0, "0101")
        time.sleep(0.2)

        self.df = DataFrame(self.kiwoom.ohlcv, columns=['open', 'high', 'low', 'close', 'volume'],
                       index=self.kiwoom.ohlcv['date'])

    def check_speedy_rising_volume(self, code):
        today = datetime.datetime.today().strftime("%Y%m%d")
        df = self.get_ohlcv(code, today)
        if df == None: return False
        volumes = df['volume']

        if len(volumes) < 21:
            return False

        sum_vol20 = 0
        today_vol = 0

        for i, vol in enumerate(volumes):
            if i == 0:
                today_vol = vol
            elif 1 <= i <= 20:
                sum_vol20 += vol
            else:
                break

        avg_vol20 = sum_vol20 / 20
        if today_vol > avg_vol20 * 10:
            return True

    def update_buy_list(self, buy_list):
        f = open("buy_list.txt", "at", encoding='utf-8')
        for code in buy_list:
            f.writelines("매수;" + code + ";시장가;10;0;매수전\n")
        f.close()

    def run(self, codes):
        codes = str(codes).zfill(6)
        buy_list = []
        num = len(codes)

        for i, code in enumerate(codes):
            print(i, '/', num)
            if self.check_speedy_rising_volume(code):
                buy_list.append(code)

        self.update_buy_list(buy_list)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pymon = PyMon()
    codes = pymon.get_code_list(sys.argv[1])
    cnt = len(codes)
    offset = int(sys.argv[2])
    if offset + 100 > cnt:
        pymon.run(codes[offset:cnt])
    else:
        pymon.run(codes[offset:offset+100])
