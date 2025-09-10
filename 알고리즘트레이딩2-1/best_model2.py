# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# 최근 1년간 3개모델의 앙상블, reinfo, pred_term, width [0, 32, 5] / 100, [1~6, 10, 15, 20, 25, 30,40], [10, 71, 5]에서
# random으로 100번 반복후 각각 평균 수익률이 최고인 reinfo, pred_term, width 선택 고정한 후 최근 2개월간 최고 기하 수익률
# 을 보이는 앙상블을 찾기 위해 randoom으로 500번 실행한 후 최근 한 달간의 best losscut을 찾음


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

    ['10HL', '15P', '5HL'],
]
random_ensemble = True


if __name__ == "__main__":

    ####################################################################################################################
    # 최근 탐색 기간에서 21개 모델 중 3개 선택, reinfo_th, pred_term, width 랜덤으로 선택, 100번 실행
    ####################################################################################################################

    if len(sys.argv) > 1:
        conf.loss_cut = float(sys.argv[1])
    else:
        conf.loss_cut = 0.005

    cnt = 100

    conf.start_time = '2024/12/01/09:00'
    conf.end_time = '2025/01/31/15:00'
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    conf.gubun = 0

    ep.mr = mr2
    ep.profit.slippage = 0.05
    margin = 10000000
    ep.margin = margin
    ep.profit.margin = margin
    add_txt = "_천만투자_slippage" + str(ep.profit.slippage)

    last_trains = [value for value in ep.last_trains if value < "2019-12-31"] + [value for value in ep.last_trains if value > "2020-12-15"]
    start_times = [value for value in ep.start_times if value < "2020/01/01/09:00"] + [value for value in ep.start_times if value > "2020/12/16/09:00"]
    end_times = [value for value in ep.end_times if value < "2020/01/15/15:00"] + [value for value in ep.end_times if value > "2020/12/31/15:00"]
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "eval_reflection_random")

    #conf.model_pools = ["5C", "5HL", "5P", "20C", "20HL", "20P", "40C", "40HL", "40P"]

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0]
    while len(results) < cnt:

        conf.reinfo_th = random.randrange(0, 32, 5) / 100#random.sample([0.2, 0.3, 0.4, 0.5], 1)[0]
        conf.pred_term = random.sample([1,2,3,4,5,6,10,15,20,25,30,40], 1)[0]#random.randint(1, 40) #random.sample([10, 20, 30, 40], 1)[0]
        conf.reinfo_width = random.randrange(10, 71, 5) #random.sample([10, 20, 30, 40, 50, 60, 70], 1)[0]
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
                conf.pred_term) + ', ' + str(conf.reinfo_width) + ']'
            if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term), str(conf.reinfo_width)] in np.array(results)[:, :4].tolist():
                continue

            dates, closes, profits, close_rates, profit_rates, MDD, profit_product, _, path = ep.main(conf, ep)


            if profit_rates[-1] > best[0]:
                best[0] = profit_rates[-1]
                best[1] = MDD
                best[2] = profit_product
                best[3] = conf.selected_model_types
                best[4] = conf.reinfo_th
                best[5] = conf.pred_term
                best[6] = conf.reinfo_width

                best_profit_rates = profit_rates
                best_ensemble = ensemble_models

                #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
                #pd.DataFrame(dic).to_excel(path, index=False)

            print("중간 best >>>>")
            print(best)

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width: ",
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term, conf.reinfo_width)

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, profit_rates[-1], MDD, profit_product])
            results_df = pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'profit_rates', 'MDD',
                                  'profit_product'])
            print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean(), len(results))
            results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    ####################################################################################################################
    # best 평균 수익률를 갖는 reinfo, pred_term, width
    ####################################################################################################################

    results_df = pd.read_csv(ep.result_path, encoding='euc-kr')

    results_df['profit_rates'] = results_df['profit_rates'].astype(float)
    results_df['self-reflection'] = results_df['self-reflection'].astype(str)

    # find best reinfo
    best = 0
    best_reinfo = 0
    for n in range(0, 31, 5):
        n = str(n / 100)
        avg = results_df.loc[results_df['self-reflection'] == n]['profit_rates'].values.mean()
        if best < avg:
            best = avg
            best_reinfo = float(n)

    # find best term
    best = 0
    best_term = 0
    for n in [1,2,3,4,5,6,10,15,20,25,30,40]:
        avg = results_df.loc[results_df['pred_term'] == n]['profit_rates'].values.mean()
        if best < avg:
            best = avg
            best_term = int(n)

    # find best width
    best = 0
    best_width = 0
    for n in range(10, 71, 5):
        avg = results_df.loc[results_df['width'] == n]['profit_rates'].values.mean()
        if best < avg:
            best = avg
            best_width = int(n)

    ####################################################################################################################
    # best reinfo, term, with를 고정시키고 최근 2개월간 best 기하 수익률 앙상블 찾기 위해 500번 실행
    ####################################################################################################################

    cnt = 500

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0]
    while len(results) < cnt:

        conf.reinfo_th = best_reinfo
        conf.pred_term = best_term
        conf.reinfo_width = best_width
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
                conf.pred_term) + ', ' + str(conf.reinfo_width) + ']'
            if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term), str(conf.reinfo_width)] in np.array(results)[:, :4].tolist():
                continue

            dates, closes, profits, close_rates, profit_rates, MDD, profit_product, _, path = ep.main(conf, ep)


            if profit_rates[-1] > best[0]:
                best[0] = profit_rates[-1]
                best[1] = MDD
                best[2] = profit_product
                best[3] = conf.selected_model_types
                best[4] = conf.reinfo_th
                best[5] = conf.pred_term
                best[6] = conf.reinfo_width

                best_profit_rates = profit_rates
                best_ensemble = ensemble_models

                #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
                #pd.DataFrame(dic).to_excel(path, index=False)

            print("중간 best >>>>")
            print(best)

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width, losscut: ",
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term,
                  conf.reinfo_width)

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width,
                            profit_rates[-1], MDD, profit_product])
            results_df = pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width',
                                  'profit_rates', 'MDD', 'profit_product'])
            print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean(), len(results))
            results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    results_df = pd.DataFrame(np.array(results),
                              columns=['ensemble', 'self-reflection', 'pred_term', 'width',
                                       'profit_rates', 'MDD', 'profit_product'])

    best_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)


    ####################################################################################################################
    # best_model.csv에 첨가할 dataframe 생성
    ####################################################################################################################
    
    import datetime
    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    best_model_path = '../알고리즘트레이딩/best_model.csv'
    df = pd.read_csv(best_model_path, encoding='euc-kr')
    add_model = []
    add_model.append([now, term1, term1,
                     '알2-1_best_model2', best_df.loc[0, 'ensemble'], best_df.loc[0, 'self-reflection'],
                      best_df.loc[0, 'pred_term'], best_df.loc[0, 'width'], best_df.loc[0, 'ensemble'],
                      best_df.loc[0, 'profit_rates'], best_df.loc[0, 'MDD'], best_df.loc[0, 'profit_product'], 0, 0])

    add_model = pd.DataFrame(np.array(add_model), columns=['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                           'ensemble', 'self-reflection', 'pred_term', 'width',
                                                           'ensemble2',	'profit_rates2', 'MDD2', 'profit_product2',
                                                           'score', 'MDD_sum'])

    ####################################################################################################################
    # bayesian_losscut으로 최근 15일 ~ 1개월 기간에서 최종 학습된 모델의 최적화된 losscut 탐색, 첨부하여 저장
    ####################################################################################################################

    import baysian_losscut as bl

    org_model = add_model.loc[0, 'ensemble2'].replace("'", "").replace('"', '').replace(']', '').replace('[', '').replace(' ', '').split(',')
    org_model[3] = float(org_model[3])
    org_model[4] = int(org_model[4])
    org_model[5] = int(org_model[5])

    bl.ep_proc = False
    bl.ep.margin = margin
    bl.ep.profit.margin = margin
    ensembles = [

        org_model

    ]
    start_time = '2025/01/01/09:00'
    end_time = '2025/01/31/15:00'
    last_train = '2025-01-15'

    bl.low = 0.001
    bl.high = 0.01

    try:
        losscut_added_model, _, _ = bl.create_losscut(ensembles, start_time, end_time, last_train)
    except:
        org_model.append("N")
        losscut_added_model = org_model
    print("best_losscut: ", str(losscut_added_model))

    # losscut을 첨부하여 모델 저장
    add_model.loc[0, 'ensemble2'] = losscut_added_model

    df = pd.concat([df, add_model], axis=0)[['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                           'ensemble', 'self-reflection', 'pred_term', 'width',
                                                           'ensemble2',	'profit_rates2', 'MDD2', 'profit_product2',
                                                           'score', 'MDD_sum']]

    df.to_csv(best_model_path, encoding='euc-kr', index=False)

    """
    # best ensembler과 지수 수익률 비교 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = "best ensemble " + best_ensemble
    plt.plot(dates, best_profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()
    """

    exit(0)