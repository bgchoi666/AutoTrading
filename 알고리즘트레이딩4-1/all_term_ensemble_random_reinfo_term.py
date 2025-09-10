# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# reinfo, pred_term은 [0, 1], [1, 40]에서 random으로 n번 반복
# 예) 2017-12-31 ~ 2021-12-31 의 trained 모델들의 앙상블 수익률 list

import data
from data import config as conf
import ensemble_proc as ep

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import sys
import datetime

import make_reinfo as mr
import make_reinfo2 as mr2
import make_reinfo3 as mr3

ensembles = [

    ["10P", "25HL", "25P"],

]
random_ensemble = True
losscut_mode = 'random'

cnt = 100


if __name__ == "__main__":

    ####################################################################################################################
    # 최근 탐색 기간에서 21개 모델 중 3개 선택, reinfo_th, pred_term, width 랜덤으로 선택, 100번 실행
    ####################################################################################################################

    if len(sys.argv) > 1:
        loss_cut = float(sys.argv[1])
    else:
        loss_cut = 0.005

    conf.start_time = '2025/06/01/09:00'
    conf.end_time = '2025/06/30/15:00'
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    conf.gubun = 0

    ep.mr = mr3
    mode = 'mr3'
    ep.profit.slippage = 0.05
    ep.margin = 10000000
    ep.profit.margin = 10000000
    if random_ensemble == False:
        add_txt = "_앙상블고정_천만투자_"+mode+"_slippage" + str(ep.profit.slippage) + losscut_mode
    else:
        add_txt = "_천만투자_" + mode + "_slippage" + str(ep.profit.slippage) + losscut_mode

    last_trains = [value for value in ep.last_trains if value < "2019-12-31"] + [value for value in ep.last_trains if value > "2020-12-15"]
    start_times = [value for value in ep.start_times if value < "2020/01/01/09:00"] + [value for value in ep.start_times if value > "2020/12/16/09:00"]
    end_times = [value for value in ep.end_times if value < "2020/01/15/15:00"] + [value for value in ep.end_times if value > "2020/12/31/15:00"]
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.conf.loss_cut = loss_cut
    ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "eval_reflection_random")

    len_terms1 = len(ep.last_trains)

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0, 0]
    while len(results) < cnt:
        # 0.04	28	17
        conf.reinfo_th = random.randrange(0, 101, 5) / 100
        if mode != 'mr3':
            conf.pred_term = random.sample([1,2,3,4,5,10,15,20,25,30,40], 1)[0]
            if conf.reinfo_th > 0.5 and conf.pred_term > 5:
                continue
        conf.reinfo_width = random.randrange(5, 71, 5)
        if losscut_mode == 'random':
            conf.loss_cut = random.randrange(10, 51, 5) / 10000
        else:
            conf.loss_cut = loss_cut
        conf.target_type = 'C'

        for enm in ensembles:

            if not random_ensemble:
                conf.selected_model_types = enm
            else:
                conf.selected_model_types = sorted(random.sample(conf.model_pools, conf.selected_num))

            ensemble_models = '["'
            for i in range(conf.selected_num - 1):
                ensemble_models = ensemble_models + conf.selected_model_types[i] + '", "'
            ensemble_models = ensemble_models + conf.selected_model_types[2] + '", ' + str(conf.reinfo_th) + ', ' + str(
                conf.pred_term) + ', ' + str(conf.reinfo_width) + ', ' + str(conf.loss_cut) + ']'
            if not results == [] and ensemble_models in np.array(results)[:, 0].tolist():
                continue

            dates, closes, profits, close_rates, profit_rates, MDD, profit_product, _, parh = ep.main(conf, ep)


            if profit_rates[-1] > best[0]:
                best[0] = profit_rates[-1]
                best[1] = MDD
                best[2] = profit_product
                best[3] = conf.selected_model_types
                best[4] = conf.reinfo_th
                best[5] = conf.pred_term
                best[6] = conf.reinfo_width
                best[7] = conf.loss_cut

                best_ensemble = ensemble_models
                best_profit_rates = profit_rates

                # dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profits_rates}
                # pd.DataFrame(dic).to_excel(path, index = False, encoding = 'euc-kr')

            print("중간 best >>>>")
            print(best, len(results))

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width, losscut: ",
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term,
                  conf.reinfo_width, conf.loss_cut)

            results.append(
                [ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut, profit_rates[-1], MDD,
                 profit_product])

            results_df = pd.DataFrame(np.array(results),
                                      columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates',
                                               'MDD', 'profit_product'])

            print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean())

            results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    results_df = pd.DataFrame(np.array(results),
                              columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD',
                                       'profit_product'])

    avg = results_df['profit_rates'].values.astype(dtype=np.float).mean()

    # profit_product순 정열후 저장
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)
    results_df.to_csv(ep.result_path, index=False)
    print('best: ', results_df['ensemble'].values[0], results_df['profit_product'].values[0])

    print("평균 >>>> ")
    print(avg)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = 'ensemble ' + best_ensemble
    plt.plot(dates, best_profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()
    exit(0)

