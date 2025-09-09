import sys
from PyQt5.QtWidgets import *
import Kiwoom
import time

class PyGran:
    def __init__(self):
        self.kiwoom = Kiwoom.Kiwoom()
        self.kiwoom.comm_connect()

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
            data.append(split_row_data[:7] + [eval])
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
    pygran.sort_buy_list()