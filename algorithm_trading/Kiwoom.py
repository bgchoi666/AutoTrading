# -*- coding:utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
#from config.kiwoomType import *
#from config.log_class import *
import pandas as pd
#import sqlite3

TR_REQ_TIME_INTERVAL = 0.2

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()

        self.calcul_data = []
        self.conditionList = []

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)
        self.OnReceiveChejanData.connect(self._receive_chejan_data)
        self.OnReceiveConditionVer.connect(self._receive_ver)
        self.OnReceiveTrCondition.connect(self._receive_condition)

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

    def get_code_list_by_market(self, market):
        code_list = self.dynamicCall("GetCodeListByMarket(QString)", market)
        code_list = code_list.split(';')
        return code_list[:-1]

    def get_master_code_name(self, code):
        code_name = self.dynamicCall("GetMasterCodeName(QString)", code)
        return code_name

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

        if rqname == "opt10001_req":
            self._opt10001(rqname, trcode)
        elif rqname == "opt10081_req":
            self._opt10081(rqname, trcode)
        elif rqname == "opw00001_req":
            self._opw00001(rqname, trcode)
        elif rqname == "opw00018_req":
            self._opw00018(rqname, trcode)
        elif rqname == "sell_req":
            self._sell(rqname, trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _receive_ver(self, IRet, sMsg):
        if(IRet == 1):
            print("조건검색식 list 불러오기 성공")
        else:
            print("조건검색식 list 불러오기 실패")
            return

        self.condCodes = ['' for i in range(16)]

        conditionNameList = self.dynamicCall('GetConditionNameList()')  # 수신된 사용자 조건검색식 리스트를 받아옴 (ex. 인덱스^조건명;)
        conditionNameListArray = conditionNameList.rstrip(';').split(';')  # 조건검색식 리스트에 마지막 ";" 기호를 삭제하고 ";" 기호 기준 분리

        self.conditionNames = []

        for i in range(len(conditionNameListArray)):
            scrNum = conditionNameListArray[i].split('^')[0]
            name = conditionNameListArray[i].split('^')[1]

            self.conditionNames.append(name)

            ret = self.dynamicCall("SendCondition(QString,QString, int, int)", scrNum, name, int(scrNum), 0)  # 선택한 조건검색을 화면번호,  조건검색 타입으로 호출
            time.sleep(0.2)

    #def _receive_condition(self, sScrNo, strCodeList, strConditionName, nIndex, nNext):
    #    print("receive_tr_condition sScrNo: " + str(sScrNo) + ", strCodeList: " + str(
    #        strCodeList) + ", strConditionName: " + str(strConditionName) + ", nIndex: " + str(
    #        nIndex) + ", nNext: " + str(nNext))  # info 레벨 로그를 남김
    #
    #    s = strCodeList.rstrip(';').split(';')
    #    self.condCodes.append(s)

    def _receive_condition(self, sScrNo, strCodeList, strConditionName, nIndex, nNext):
        idx = self.conditionNames.index(strConditionName)
        print("receive_tr_condition sScrNo: " + str(sScrNo) + ", strCodeList: " + str(
            strCodeList) + ", strConditionName: " + str(strConditionName) + ", nIndex: " + str(
            nIndex) + ", nNext: " + str(nNext))  # info 레벨 로그를 남김

        s = strCodeList.rstrip(';').split(';')
        self.condCodes[idx] = s

    def _opt10001(self, rqname, trcode):

        self.opt10001_종목코드 = self._comm_get_data(trcode, "", rqname, 0, "종목코드")
        self.opt10001_종목명 = self._comm_get_data(trcode, "", rqname, 0, "종목명")
        self.opt10001_현재가 = self._comm_get_data(trcode, "", rqname, 0, "현재가")
        self.opt10001_연중최고 = self._comm_get_data(trcode, "", rqname, 0, "연중최고")
        self.opt10001_연중최저 = self._comm_get_data(trcode, "", rqname, 0, "연중최저")
        self.opt10001_PER = self._comm_get_data(trcode, "", rqname, 0, "PER")
        self.opt10001_PBR = self._comm_get_data(trcode, "", rqname, 0, "PBR")
        self.opt10001_ROE = self._comm_get_data(trcode, "", rqname, 0, "ROE")


    def _opt10081(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)

        for i in range(data_cnt):
            date = self._comm_get_data(trcode, "", rqname, i, "일자")
            open = self._comm_get_data(trcode, "", rqname, i, "시가")
            high = self._comm_get_data(trcode, "", rqname, i, "고가")
            low = self._comm_get_data(trcode, "", rqname, i, "저가")
            close = self._comm_get_data(trcode, "", rqname, i, "현재가")
            volume = self._comm_get_data(trcode, "", rqname, i, "거래량")

            self.ohlcv['date'].append(date)
            self.ohlcv['open'].append(int(open))
            self.ohlcv['high'].append(int(high))
            self.ohlcv['low'].append(int(low))
            self.ohlcv['close'].append(int(close))
            self.ohlcv['volume'].append(int(volume))

    def _opw00018(self, rqname, trcode):
        # single data
        total_purchase_price = self._comm_get_data(trcode, "", rqname, 0, "총매입금액")
        total_eval_price = self._comm_get_data(trcode, "", rqname, 0, "총평가금액")
        total_eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, 0, "총평가손익금액")
        total_earning_rate = self._comm_get_data(trcode, "", rqname, 0, "총수익률(%)")
        estimated_deposit = self._comm_get_data(trcode, "", rqname, 0, "추정예탁자산")

        self.opw00018_output['single'].append(Kiwoom.change_format(total_purchase_price))
        self.opw00018_output['single'].append(Kiwoom.change_format(total_eval_price))
        self.opw00018_output['single'].append(Kiwoom.change_format(total_eval_profit_loss_price))

        total_earning_rate = Kiwoom.change_format(total_earning_rate)

        if self.get_server_gubun():
            total_earning_rate = float(total_earning_rate) / 100
            total_earning_rate = str(total_earning_rate)

        self.opw00018_output['single'].append(total_earning_rate)

        #self.opw00018_output['single'].append(Kiwoom.change_format(total_earning_rate))
        self.opw00018_output['single'].append(Kiwoom.change_format(estimated_deposit))

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        for i in range(rows):
            name = self._comm_get_data(trcode, "", rqname, i, "종목명")
            quantity = self._comm_get_data(trcode, "", rqname, i, "보유수량")
            purchase_price = self._comm_get_data(trcode, "", rqname, i, "매입가")
            current_price = self._comm_get_data(trcode, "", rqname, i, "현재가")
            eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, i, "평가손익")
            earning_rate = self._comm_get_data(trcode, "", rqname, i, "수익률(%)")
            code = self._comm_get_data(trcode, "", rqname, i, "종목번호")

            quantity = Kiwoom.change_format(quantity)
            purchase_price = Kiwoom.change_format(purchase_price)
            current_price = Kiwoom.change_format(current_price)
            eval_profit_loss_price = Kiwoom.change_format(eval_profit_loss_price)
            earning_rate = str(float(Kiwoom.change_format2(earning_rate))/100)

            self.opw00018_output['multi'].append([name, quantity, purchase_price, current_price,
                                                  eval_profit_loss_price, earning_rate, code])

    def _sell(self, rqname, trcode):

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        for i in range(rows):
            code = self._comm_get_data(trcode, "", rqname, i, "종목번호")
            code = code[1:]

            earning_rate = self._comm_get_data(trcode, "", rqname, i, "수익률(%)")
            earning_rate = Kiwoom.change_format2(earning_rate)

            num = self._comm_get_data(trcode, "", rqname, i, "보유수량")
            num = Kiwoom.change_format(num)

            self.sell_output['multi'].append([code, earning_rate, num])

    def _opw00001(self, rqname, trcode):
        self.d2_deposit = self._comm_get_data(trcode, "", rqname, 0, "d+2추정예수금")
        #self.d2_deposit = Kiwoom.change_format(d2_deposit)

    def send_order(self, rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no):
        self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                         [rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no])

    def get_chejan_data(self, fid):
        ret = self.dynamicCall("GetChejanData(int)", fid)
        return ret

    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        print(gubun)
        print(self.get_chejan_data(9203))
        print(self.get_chejan_data(302))
        print(self.get_chejan_data(900))
        print(self.get_chejan_data(901))

    def get_login_info(self, tag):
        ret = self.dynamicCall("GetLoginInfo(QString)", tag)
        return ret


    def reset_opw00018_output(self):
        self.opw00018_output = {'single': [], 'multi': []}

    def reset_sell_output(self):
        self.sell_output = {'single': [], 'multi': []}

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

    def get_server_gubun(self):
        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()

    account_number = kiwoom.get_login_info("ACCNO")
    print(account_number)
    account_number = account_number.split(';')[1]
    print(account_number)

    kiwoom.set_input_value("계좌번호", account_number)
    kiwoom.set_input_value("비밀번호", "0000")
    kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")
