# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 거래 내역 trans.csv에 저장

import pandas as pd
import logging
import os
import shutil
import datetime

logging.basicConfig(filename='futureTrader60M.log', level=logging.DEBUG)

def save_trans(gubun, purchase_num, purchase_price):

    pred_date = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    # 주문 내역 거래내역 파일 (trans.csv) 에 저장
    trans_df = pd.read_csv('trans.csv', encoding='euc-kr')
    거래일시 = trans_df.loc[trans_df.index.max(), '거래일시']
    청산가격 = int(trans_df.loc[trans_df.index.max(), '청산가격'])
    매매구분 = trans_df.loc[trans_df.index.max(), '매매구분']
    수량 = int(trans_df.loc[trans_df.index.max(), '수량'])
    거래가격 = float(trans_df.loc[trans_df.index.max(), '거래가격'])

    if 청산가격 != 0 or 거래일시[:10] != pred_date[:10]:

        # 신규 주문 내역 거래내역 파일 (trans.csv) 에 저장
        trans_data = {'거래일시': pred_date, '매매구분': gubun, '수량': purchase_num,
                      '거래가격': purchase_price, '청산일시': '0', '청산가격': '0'}
        trans_df = trans_df.append(trans_data, ignore_index=True)

    elif 매매구분 == gubun:

        # 추가 주문 내역 거래내역 파일 (trans.csv) 에 저장
        trans_df.loc[trans_df.index.max(), '거래일시'] = pred_date
        trans_df.loc[trans_df.index.max(), '수량'] = 수량 + purchase_num
        trans_df.loc[trans_df.index.max(), '거래가격'] = (수량*거래가격 + purchase_num*purchase_price) / (수량 + purchase_num)

    elif purchase_num > 수량:

        # 청산후 신규 주문
        trans_df.loc[trans_df.index.max(), '청산일시'] = pred_date
        trans_df.loc[trans_df.index.max(), '청산가격'] = purchase_price
        trans_df = trans_df.append({'거래일시': pred_date, '매매구분': gubun, '수량': int(purchase_num - 수량),
                                    '거래가격': purchase_price, '청산일시': '0', '청산가격': '0'}, ignore_index=True)

    elif purchase_num < 수량:

        # 일부 청산후 수정
        trans_df.loc[trans_df.index.max(), '청산일시'] = pred_date
        trans_df.loc[trans_df.index.max(), '수량'] = purchase_num
        trans_df.loc[trans_df.index.max(), '청산가격'] = purchase_price
        trans_df = trans_df.append({'거래일시': pred_date, '매매구분': 매매구분, '수량': int(수량 - purchase_num),
                                    '거래가격': 거래가격, '청산일시': '0', '청산가격': '0'}, ignore_index=True)

    else:

        # 모두 청산
        trans_df.loc[trans_df.index.max(), '청산일시'] = pred_date
        trans_df.loc[trans_df.index.max(), '청산가격'] = purchase_price


    trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')


    print(trans_df.values[-1])

    # 주문 내역 log 파일에 저장
    logging.info("주문내역")
    logging.info(trans_df[-1:])
    now = datetime.datetime.now()
    logging.info(now)

    """
    pred_date = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    
    # 주문 내역 거래내역 파일 (trans.csv) 에 저장
    trans_df = pd.read_csv('trans.csv', encoding='euc-kr')
    trans_data = {'거래일시': pred_date, '매매구분': gubun, '수량': purchase_num,
                  '거래가격': purchase_price, '청산일시': '0', '청산가격': '0'}
    trans_df = trans_df.append(trans_data, ignore_index=True)
    trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')
    print(trans_data)
    
    # 주문 내역 log 파일에 저장
    logging.info("주문내역")
    logging.info(trans_data)
    now = datetime.datetime.now()
    logging.info(now)
    
    # temporary 개별 거래 파일 본 파일로 move
    if os.path.isfile('trans_2_temp.csv'): shutil.move('trans_2_temp.csv', 'trans_2.csv')
    if os.path.isfile('trans_4_temp.csv'): shutil.move('trans_4_temp.csv', 'trans_4.csv')
    if os.path.isfile('trans_new_temp.csv'): shutil.move('trans_new_temp.csv', 'trans_new.csv')
    
    print('신규 ' + gubun + ' 주문 완료... trans.csv 파일에 주문 내역 저장...')
    """

def save_liquidation(price):

    # 청산 내역 통합 trans file에 저장
    trans_df = pd.read_csv("trans.csv", encoding="euc-kr")
    trading_date = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    trans_df.loc[trans_df.index.max(), '청산일시'] = trading_date
    trans_df.loc[trans_df.index.max(), '청산가격'] = price

    trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')

    print('청산내역 trnas.csv에 저장')
    print(datetime.datetime.now())
    logging.info(datetime.datetime.now())
    logging.info('청산내역 trnas.csv에 저장')
    logging.info(trans_df[-1:])


    """
    # 청산 내역 통합 trans file에 저장
    trans_df = pd.read_csv("trans.csv", encoding="euc-kr")
    if type == "매수청산":
        trans_df = trans_df.append(
            {'거래일시': trading_date, '매매구분': '매도', '수량': holding_num, '거래가격': purchase_price, '청산일시': trading_date,
             '청산가격': current_price},
            ignore_index=True)
    elif type == '매도청산':
        trans_df = trans_df.append(
            {'거래일시': trading_date, '매매구분': '매수', '수량': holding_num, '거래가격': purchase_price, '청산일시': trading_date,
             '청산가격': current_price},
            ignore_index=True)
    else:
        return False
    
    trans_df.to_csv('trans.csv', index=False, encoding='euc-kr')
    print('청산내역 trnas.csv에 저장')
    
    logging.info(datetime.datetime.now())
    logging.info('청산내역 trnas.csv에 저장')
    logging.info(trans_df[-1:])
    """