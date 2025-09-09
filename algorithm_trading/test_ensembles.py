# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기존 앙상블 파일로부터 각 앙상블에 대한 주어진 구간에서의 예측 수익률 첨가

import pandas as pd
import ensemble_test as ep
import sys

ep.selected_num = 3

ep.last_train = '2021-12-24'

ep.start_time = '2021/12/27/09:00'
ep.end_time = '2022/01/05/15:00'

ep.checkpoint_path = ['' for i in range(22)]
ep.checkpoint_path[1] = ep.last_train+"/60M_5C_best"
ep.checkpoint_path[2] = ep.last_train+"/60M_5HL_best"
ep.checkpoint_path[3] = ep.last_train+"/60M_5P_best"
ep.checkpoint_path[4] = ep.last_train+"/60M_10C_best"
ep.checkpoint_path[5] = ep.last_train+"/60M_10HL_best"
ep.checkpoint_path[6] = ep.last_train+"/60M_10P_best"
ep.checkpoint_path[7] = ep.last_train+"/60M_15C_best"
ep.checkpoint_path[8] = ep.last_train+"/60M_15HL_best"
ep.checkpoint_path[9] = ep.last_train+"/60M_15P_best"
ep.checkpoint_path[10] = ep.last_train+"/60M_20C_best"
ep.checkpoint_path[11] = ep.last_train+"/60M_20HL_best"
ep.checkpoint_path[12] = ep.last_train+"/60M_20P_best"
ep.checkpoint_path[13] = ep.last_train+"/60M_25C_best"
ep.checkpoint_path[14] = ep.last_train+"/60M_25HL_best"
ep.checkpoint_path[15] = ep.last_train+"/60M_25P_best"
ep.checkpoint_path[16] = ep.last_train+"/60M_30C_best"
ep.checkpoint_path[17] = ep.last_train+"/60M_30HL_best"
ep.checkpoint_path[18] = ep.last_train+"/60M_30P_best"
ep.checkpoint_path[19] = ep.last_train+"/60M_40C_best"
ep.checkpoint_path[20] = ep.last_train+"/60M_40HL_best"
ep.checkpoint_path[21] = ep.last_train+"/60M_40P_best"

ep.selected_checkpoint_path = ['' for i in range(ep.selected_num)]
ep.weights = [1 for i in range(ep.selected_num)]

file_name = ep.last_train+"/ensemble_results.csv"

ep.df0_path = 'kospi200f_11_60M.csv'
ep.df_pred_path = ep.last_train+'/kospi200f_60M_pred.csv'
ep.result_path = ep.last_train+'/pred_83_results.csv'

def run():
    ens = pd.read_csv(file_name, encoding='euc-kr')

    rate = []
    cnt = 0
    for i in range(len(ens)):
        e = ens.values[i, :ep.selected_num]

        for j in range(ep.selected_num):
            ep.selected_checkpoint_path[j] = ep.checkpoint_path[int(e[j])]

        if i == 0:
            ep.preprocessing()
        p = str(ep.predict())
        rate.append(p)

        print(e)
        print(p)

        cnt += 1
        if cnt % 100 == 0:
            print("========================================================================")
            print(" ")
            print(cnt)
            print(" ")
            print("========================================================================")
    ens[ep.start_time[5:10]+'~'+ep.end_time[5:10]] = rate

    ens.to_csv(file_name, index=False, encoding='euc-kr')

if __name__ == "__main__":
    run()
    sys.exit(0)