# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 모델 타입을 생성하여 주어진 구간에서 train, test, prediction
# - 현재 종가가 다음 n일 동안 최고가 이면 0, 최저가 이면 1
# - 중립 state는 없음
# DMI, stocastic은 non-normalization

import data
from data import config as conf
import model

import datetime
import sys

conf.gubun = 0 # 0:predict only 1:test only 2:train

conf.pred_term = 5
conf.target_type = 'HL'

conf.last_train = '2023-08-15'
conf.start_time = '2023/08/16/09:00'
conf.end_time = '2023/08/30/15:00'

# file path 설정
data.set_path(conf)

data.set_profit(conf, 0.005, 1) # loss_cut, profit_cut


if __name__ == "__main__":
    if len(sys.argv) < 2:
       conf.gubun = 0
    else:
        conf.gubun = int(sys.argv[1])

        conf.pred_term, conf.target_type, conf.base1, conf.base2 = model.parse(sys.argv[2])

        if len(sys.argv) >= 6:

            conf.last_train = sys.argv[3]
            conf.start_time = sys.argv[4]
            conf.end_time = sys.argv[5]

            data.set_path(conf)

        else:
            print('training, test date error!')
            exit(1)

    m = model.create_model(conf)

    if conf.gubun == 0:
        print('data processing start...')
        now = datetime.datetime.now()
        print(now)
        data.preprocessing(conf)
        print('data processing end...')

        m.load_weights(conf.checkpoint_path_best)
        r = model.predict(m, conf)

        import profit
        profit.pred_term =conf.pred_term
        profit.result_path = conf.result_path
        p = profit.calc_profit()
        print(conf.last_train + " 수익률: " + str(p))

    elif conf.gubun == 1:
        now = datetime.datetime.now()
        print(now)
        m.load_weights(conf.checkpoint_path_best)
        model.test(m, conf)

    else:
        print('data processing start...')
        now = datetime.datetime.now()
        print(now)
        data.preprocessing(conf)
        print('data processing end...')

        data.make_train_data(conf)

        now = datetime.datetime.now()
        print(now)
        model.train(m, conf)
        print('training end...')

        now = datetime.datetime.now()
        print(now)
        model.test(m, conf)

        r = model.predict(m, conf)
        #print(r)

        import profit
        profit.loss_cut = conf.loss_cut
        profit.pred_term = conf.pred_term
        profit.result_path = conf.result_path
        p = profit.calc_profit()
        print(conf.last_train + " 수익률: " + str(p))