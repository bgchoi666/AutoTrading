# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 구간에서 train된 모델들에 대한 앙상블들을 생성하여 각각의 수익률과 함께 저장

import ensemble_test as ep
import datetime
import pandas as pd
import numpy as np
import sys
import random

ep.selected_num = 3

ep.start_time = '2022/07/16/09:00'
ep.end_time = '2022/07/31/15:00'
ep.last_train = '2022-07-15'

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

ep.df0_path = 'kospi200f_11_60M.csv'
ep.df_pred_path = ep.last_train+'/kospi200f_60M_pred.csv'
ep.result_path = ep.last_train+'/pred_83_results.csv'

def create_ensembles():
    print(datetime.datetime.now())
    results = []
    total_ensembles = []
    for i in range(1330):
        ensembles = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], ep.selected_num)
        while sorted(ensembles) in total_ensembles:
            ensembles = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], ep.selected_num)
        total_ensembles.append(sorted(ensembles))

        ep.selected_checkpoint_path = ['' for i in range(ep.selected_num)]
        for j in range(ep.selected_num):
            ep.selected_checkpoint_path[j] = ep.checkpoint_path[ensembles[j]]

        if i == 0:
            ep.preprocessing()
        ensembles.append(ep.predict())
        print(ensembles)
        results.append(ensembles)

        if i % 100 == 0:
            print("========================================================================")
            print(" ")
            print(i)
            print(" ")
            print("========================================================================")

    print(datetime.datetime.now())
    pd.DataFrame(np.array(results), columns=['0', '1', '2', ep.start_time[5:10]+'~'+ep.end_time[5:10]]).to_csv(ep.last_train+"/ensemble_results.csv", index=False, encoding='euc-kr')

    models = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    df = []

    selected_models = get_best_ensemble(ep.last_train+"/ensemble_results.csv", models)
    print('best ensemble')
    print(selected_models)
    selected_models.append('kospi200f')
    df.append(selected_models)

    pd.DataFrame(np.array((df))).to_csv(ep.last_train+"/best_model.csv", index=False, encoding='euc-kr')

def get_best_ensemble(ensemble_path, models):
    selected_models = ["" for i in range(ep.selected_num)]
    df = pd.read_csv(ensemble_path, encoding='euc-kr')
    df = df.sort_values(df.columns[-1], ascending=False)
    freq = [0 for i in range(len(models))]
    for i in range(10):
        for j in range(ep.selected_num):
            freq[models.index(df.values[i, j])] += 1
    for k in range(ep.selected_num):
        max_idx = np.argmax(np.array(freq))
        selected_models[k] = models[max_idx]
        freq[max_idx] = 0

    return selected_models

if __name__ == "__main__":

    if len(sys.argv) > 1:
        ep.last_train = sys.argv[1]
        ep.start_time = sys.argv[2]
        ep.end_time = sys.argv[3]

        ep.df_pred_path = ep.last_train + '/kospi200f_60M_pred.csv'
        ep.result_path = ep.last_train + '/pred_83_results.csv'

    create_ensembles()
    sys.exit(0)