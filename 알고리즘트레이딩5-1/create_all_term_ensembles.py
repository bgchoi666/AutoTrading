# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 구간에서 train된 모델들에 대한 앙상블들을 생성하여 각각의 수익률과 함께 저장

from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import sys
import os
import datetime
import random


def create_ensembles():

    columns = ['0', '1', '2', 'profit_rates', 'MDD', 'profit_product']

    models = conf.model_pools
    cnt = 100

    if os.path.isfile(ep.result_path):
        results = pd.read_csv(ep.result_path).values.tolist()
        total_ensembles = pd.read_csv(ep.result_path).values[:, :3].tolist()
    else:
        results = []
        total_ensembles = []


    best_profit_rates = 0
    best_ensemble = ''
    for i in range(cnt):
        print(datetime.datetime.now())
        ensembles = random.sample(models, conf.selected_num)
        while sorted(ensembles) in total_ensembles:
            ensembles = random.sample(models, conf.selected_num)
        ensembles = sorted(ensembles)
        total_ensembles.append(ensembles)

        conf.selected_model_types = ensembles

        dates, closes, profits, close_rates, profits_rates, MDD, profit_product, ensemble_models, path = ep.main(conf, ep)

        if profits_rates[-1] > best_profit_rates:
            best_profit_rates = profits_rates[-1]
            best_ensemble = ensemble_models

            dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profits_rates}
            pd.DataFrame(dic).to_excel(path, index=False, encoding='euc-kr')

        print("best profit rates: " + str(best_profit_rates) + ", " + best_ensemble)

        print(ensembles, profits_rates[-1], MDD, profit_product)
        results.append([ensembles[0], ensembles[1], ensembles[2], profits_rates[-1], MDD, profit_product])

        if i % 10 == 0:
            print("========================================================================")
            print(" ")
            print(i)
            print(" ")
            print("========================================================================")

        print(datetime.datetime.now())
        pd.DataFrame(np.array(results), columns=columns).drop_duplicates().to_csv(ep.result_path, index=False, encoding='euc-kr')

    print('...ending', datetime.datetime.now())
    pd.DataFrame(np.array(results), columns=columns).drop_duplicates().to_csv(ep.result_path, index=False, encoding='euc-kr')

if __name__ == "__main__":

    conf.start_time = '2017/01/01/09:00'
    conf.end_time = '2022/12/31/15:00'

    conf.reinfo_th = 0.06
    conf.pred_term = 39
    conf.reinfo_width = 36
    conf.gubun = 0

    import make_reinfo2 as mr2
    ep.mr = mr2

    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times, ep.end_times,
		 "_reinfo2_" + str(conf.reinfo_th) + "_term_" + str(conf.pred_term) + "_width_" + str(conf.reinfo_width))
    ep.folder, ep.result_path = ep.set_path("best_ensemble", "all_ensemble_results")

    create_ensembles()
    sys.exit(0)