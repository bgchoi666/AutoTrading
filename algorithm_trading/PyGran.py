import sys
from PyQt5.QtWidgets import *
import Kiwoom
import time
import pandas as pd
from pandas import DataFrame
import datetime
import random
#from config.kiwoomType import *
#from config.log_class import *


MARKET_KOSPI   = 0
MARKET_KOSDAQ  = 10

class PyGran:
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

    def granville_signal(self, code, df):

        name = self.kiwoom.get_master_code_name(code)
        cnt = len(df.values)
        print(name)

        print("총 일수 %d" % cnt)

        pass_success = False

        # 120일 이평선을 그릴만큼의 데이터가 있는지 체크
        if cnt < 120:
            pass_success = False

        else:

            # 120일 이평선의 최근 가격 구함
            total_price = 0
            for value in df['close'].values[:120]:
                total_price += int(value)
            moving_average_price = total_price / 120

            # 오늘자 주가가 120일 이평선에 걸쳐있는지 확인
            bottom_stock_price = False
            check_price = None
            if int(df['low'].values[0]) <= moving_average_price and moving_average_price <= int(
                    df['high'].values[0]):
                print("오늘 주가 120이평선 아래에 걸쳐있는 것 확인")
                bottom_stock_price = True
                check_price = int(df['high'].values[0])

            # 과거 일봉 데이터를 조회하면서 120일 이평선보다 주가가 계속 밑에 존재하는지 확인
            prev_price = None
            if bottom_stock_price == True:

                moving_average_price_prev = 0
                price_top_moving = False
                idx = 1
                while True:

                    if len(df.values[idx:]) < 120:  # 120일치가 있는지 계속 확인
                        print("120일치가 없음")
                        break

                    total_price = 0
                    for value in df['close'].values[idx:120 + idx]:
                        total_price += int(value)
                    moving_average_price_prev = total_price / 120

                    if moving_average_price_prev <= int(df['high'].values[idx]) and idx <= 20:
                        print("20일 동안 주가가 120일 이평선과 같거나 위에 있으면 조건 통과 못함")
                        price_top_moving = False
                        break

                    elif int(df['low'].values[idx]) > moving_average_price_prev and idx > 20:  # 120일 이평선 위에 있는 구간 존재
                        print("120일치 이평선 위에 있는 구간 확인됨")
                        price_top_moving = True
                        prev_price = int(df['low'].values[idx])
                        break

                    idx += 1

                # 해당부분 이평선이 가장 최근의 이평선 가격보다 낮은지 확인
                if price_top_moving == True:
                    if moving_average_price > moving_average_price_prev and check_price > prev_price:
                        print("포착된 이평선의 가격이 오늘자 이평선 가격보다 낮은 것 확인")
                        print("포착된 부분의 저가가 오늘자 주가의 고가보다 낮은지 확인")
                        pass_success = True
        if pass_success: print(name + ":조건 통과")
        else: print(name + ":조건 통과 못함")

        return pass_success

    def new_high(self, code, df):

        name = self.kiwoom.get_master_code_name(code)
        cnt = len(df.values)
        print(name)

        print("총 일수 %d" % cnt)

        pass_success = False

        # 120일 종가 존재 여부 확인
        if cnt < 251:
            pass_success = False
            print("거래일수 120일 미만")

        else:

            # 120일 고가 계산
            prev_high = max((df['high'].values[1:251]))

            # 오늘자 주가가 신고가 갱신했는지 확인
            if df['close'].values[0] > prev_high:
                new_high_price = True
            else: new_high_price = False

            # 120일 평균 거래량 계산
            avg_volume = sum((df['volume'].values[1:251]))/250

            # 오늘자 거래량이 120일 평균 거래량보다 1000% 이상 급증했는지 확인
            if df['volume'].values[0] > avg_volume * 10:
                upsurge_volume = True
            else: upsurge_volume = False

            if new_high_price and upsurge_volume:
                pass_success = True
                print("250일 신고가 and 거래량 급증")

        if pass_success: print(name + ":조건 통과")
        else: print(name + ":조건 통과 못함")

        return pass_success

    def moving_average_break(self, code, df):

        name = self.kiwoom.get_master_code_name(code)
        cnt = len(df['close'])

        print(name + ": 20일 이동평균 상향 돌파시 매수")
        print("남은 일자 수 %s" % cnt)
        print("총 일수 %s" % len(self.calcul_data))

        pass_success = False

        # 20일 이평선을 그릴만큼의 데이터가 있는지 체크
        if len(df.values) < 20:
            pass_success = False

        else:
            data = df.values
            # 20일 이평선의 최근 가격 구함
            total_price = 0
            for value in df['close'].values[:20]:
                total_price += int(value)
            moving_average_price_20 = total_price / 20

            # 5일 이평선의 최근 가격 구함
            total_price = 0
            for value in df['close'].values[:5]:
                total_price += int(value)
            moving_average_price_5 = total_price / 5

            """
            # 오늘자 주가가 20일 이평선을 돌파하는지 확인
            bottom_stock_price = False
            if moving_average_price < int(df['close'].values[0]):
                print("오늘 주가 20이평선 돌파 확인")
                bottom_stock_price = True

            # 과거 일봉 데이터를 조회하면서 3일 연속 20일 이평선보다 주가가 계속 밑에 존재하는지 확인
            if bottom_stock_price == True:

                moving_average_price_prev = 0
                price_bottom_moving = True
                idx = 1
                while idx <= 1:

                    if len(df.values[idx:]) < 20:  # 20일치가 있는지 계속 확인
                        print("20일치가 없음")
                        price_bottom_moving = False
                        break

                    total_price = 0
                    for value in df['close'].values[idx:20 + idx]:
                        total_price += int(value)
                    moving_average_price_prev = total_price / 20

                    if moving_average_price_prev < int(df['close'].values[idx]):
                        print("3일 동안 주가가 20일 이평선과 같거나 위에 있으면 조건 통과 못함")
                        price_bottom_moving = False
                        break

                    idx += 1
            """
            # 5일 이평선이 20일 이평선을 돌파하는지 확인
            if moving_average_price_5 > moving_average_price_20:
                print("5일 이평선이 20일 이평선 돌파 확인")
                pass_success = True
        if pass_success: print(name + ": 20일 이평 상향 돌파 조건 통과")
        else: print(name + ": 20일 이평 상향 돌파 조건 통과 못함")

        return pass_success

    def get_ohlcv(self, code, start):
        self.kiwoom.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("기준일자", start)
        self.kiwoom.set_input_value("수정주가구분", 1)
        self.kiwoom.comm_rq_data("opt10081_req", "opt10081", 0, "0101")
        time.sleep(0.2)

        df = DataFrame(self.kiwoom.ohlcv, columns=['open', 'high', 'low', 'close', 'volume'],
                       index=self.kiwoom.ohlcv['date'])
        return df

    def update_buy_list(self, buy_list):
        f = open("buy_list.txt", "at", encoding='utf-8')
        for code in buy_list:
            name = self.kiwoom.get_master_code_name(code)
            f.writelines("매수;" + code + ";시장가;10;0;매수전;" + name + "\n")
        f.close()

    def run(self, gubun, codes):
        buy_list = []
        num = len(codes)

        for i, code in enumerate(codes):
            code = str(code).zfill(6)
            print(i, '/', num)

            today = datetime.datetime.today().strftime("%Y%m%d")
            df = self.get_ohlcv(code, today)

            if gubun == 'granville':
                if self.granville_signal(code, df):
                    buy_list.append(code)
            elif gubun == 'mabreak':
                if self.moving_average_break(code, df):
                    buy_list.append(code)
            elif gubun == 'newHigh':
                if self.new_high(code, df):
                    buy_list.append(code)

        self.update_buy_list(buy_list)

    def sort_buy_list(self):
        f = open("buy_list.txt", 'rt', encoding='utf-8')
        buy_list = f.read().splitlines()
        f.close()

        data = []
        for row_data in buy_list:
            split_row_data = row_data.split(';')
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

# argument 1: market selection, 'kospi', 'kosdaq', 'kospi200'  argumnet 2: offset
# argument 3: serch method  'granville', 'newHigh', 'mabreak'
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pygran = PyGran()
    codes = pygran.get_code_list(sys.argv[1])
    cnt = len(codes)
    offset = int(sys.argv[2])
    if offset + 100 > cnt:
        pygran.run(sys.argv[3], codes[offset:cnt])
    else:
        pygran.run(sys.argv[3], codes[offset:offset+100])