# Copyright 2025 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# reinfo, pred_term, losscut은 [0, 0.3, 0.05], [1~5] and [5, 41, 5], [0.001, 0.01, 0.005]에서 random으로 n번 반복
# best model 선택후 다시 앙상블 모델 고정후 n번 반복하영 best reinof, pred_term, width, losscut 결정

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

ensembles = [

    ['10HL', '15P', '5HL'],
]

# 기본 변수 설정
conf.gubun = 0

ep.mr = mr2
ep.profit.slippage = 0.05
margin = 10000000
ep.margin = margin
ep.profit.margin = margin

conf.selected_num = 3

if __name__ == "__main__":

    ####################################################################################################################
    # 최근 탐색 기간에서 21개 모델 중 3개 선택, reinfo_th, pred_term, width 랜덤으로 선택, n번 실행
    ####################################################################################################################

    random_ensemble = True
    losscut_mode = ''

    cnt = 100

    if len(sys.argv) > 1:
        conf.loss_cut = float(sys.argv[1])
    else:
        conf.loss_cut = 0.005


    start_time = '2024/10/01/09:00'
    end_time = '2024/12/31/15:00'
    conf.start_time = start_time
    conf.end_time = end_time
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    add_txt = "_천만투자_slippage" + str(ep.profit.slippage) + losscut_mode

    last_trains = [value for value in ep.last_trains if value < "2019-12-31"] + [value for value in ep.last_trains if value > "2020-12-15"]
    start_times = [value for value in ep.start_times if value < "2020/01/01/09:00"] + [value for value in ep.start_times if value > "2020/12/16/09:00"]
    end_times = [value for value in ep.end_times if value < "2020/01/15/15:00"] + [value for value in ep.end_times if value > "2020/12/31/15:00"]
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "eval_reflection_random")

    len_terms1 = len(ep.last_trains)

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0, 0]
    while len(results) < cnt:

        conf.reinfo_th = random.randrange(0, 31, 5) / 100#random.sample([0.2, 0.3, 0.4, 0.5], 1)[0]
        conf.pred_term = random.sample([1,2,3,4,5,6,10,15,20,25,30,40], 1)[0]#random.randint(1, 40) #random.sample([10, 20, 30, 40], 1)[0]
        conf.reinfo_width = 100#random.randrange(10, 71, 5) #random.sample([10, 20, 30, 40, 50, 60, 70], 1)[0]
        if losscut_mode == 'random':
            conf.loss_cut = random.randrange(10, 51, 5) / 10000 # random.sample([10, 20, 30, 40, 50, 60, 70], 1)[0]
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
            if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term), str(conf.reinfo_width),
                                      str(conf.loss_cut)] in np.array(results)[:, :5].tolist():
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
                best[7] = conf.loss_cut

                best_profit_rates = profit_rates
                best_ensemble = ensemble_models

                #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
                #pd.DataFrame(dic).to_excel(path, index=False)

            print("중간 best >>>>")
            print(best)

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width, losscut: ",
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th,
                  conf.pred_term, conf.reinfo_width, conf.loss_cut)

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                            profit_rates[-1], MDD, profit_product])
            results_df = pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                  'profit_rates', 'MDD', 'profit_product'])
            print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean(), len(results))
            results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    results_df = pd.DataFrame(np.array(results),
                              columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                       'profit_rates', 'MDD', 'profit_product'])

    ####################################################################################################################
    # 기하 수익률이 제일 높은 앙상블 선택
    ####################################################################################################################

    # result dataframe profit_product 내림차순 best 선택
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)
    top_ensemble = results_df['ensemble'].values[0]
    top_reino = float(results_df['self-reflection'].values[0])
    top_term = int(results_df['pred_term'].values[0])
    top_width = int(results_df['width'].values[0])
    top_losscut = float(results_df['losscut'].values[0])
    top_rate = float(results_df['profit_rates'].values[0])
    top_MDD = float(results_df['MDD'].values[0])
    top_product = float(results_df['profit_product'].values[0])

    ####################################################################################################################
    # 앙상블 고정 후 나머지 변수 들에 대하여 random으로 n번 반복후 best 기하 수익률 모델 선택
    ####################################################################################################################

    top_ensemble = top_ensemble.replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
    ensembles = [top_ensemble[:3]]

    random_ensemble = False
    losscut_mode = 'random'

    cnt = 225

    add_txt = "_천만투자_slippage" + str(ep.profit.slippage) + "_best_enm_" + losscut_mode

    conf.start_time = start_time
    conf.end_time = end_time
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "eval_reflection_random")

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0, 0]
    while len(results) < cnt:

        conf.reinfo_th = random.randrange(max(0, top_reino*100 - 5*2), min(100, top_reino*100 + 5*2) + 1, 5) / 100
        term_list = [1,2,3,4,5,6,10,15,20,25,30,40]
        term_idx = np.where(np.array(term_list) == top_term)[0][0]
        conf.pred_term = random.sample(term_list[max(0, term_idx-2):min(11, term_idx+2) + 1], 1)[0]
        conf.reinfo_width = 100#random.randrange(min(10, top_width - 5), max(70, top_width + 5) + 1, 5)
        if losscut_mode == 'random':
            conf.loss_cut = random.randrange(10, 51, 5) / 10000
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
            if not results == [] and [ensemble_models, str(conf.reinfo_th), str(conf.pred_term), str(conf.reinfo_width),
                                      str(conf.loss_cut)] in np.array(results)[:, :5].tolist():
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
                best[7] = conf.loss_cut

                best_profit_rates = profit_rates
                best_ensemble = ensemble_models

                #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
                #pd.DataFrame(dic).to_excel(path, index=False)

            print("중간 best >>>>")
            print(best)

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width, losscut: ",
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th,
                  conf.pred_term, conf.reinfo_width, conf.loss_cut)

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                            profit_rates[-1], MDD, profit_product])
            results_df = pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                  'profit_rates', 'MDD', 'profit_product'])
            print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean(), len(results))
            results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    results_df = pd.DataFrame(np.array(results),
                              columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                       'profit_rates', 'MDD', 'profit_product'])

    ####################################################################################################################
    # 기하 수익률이 최고인 모델 선택 저장
    ####################################################################################################################
    
    # result dataframe profit_product 내림차순 best 선택
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)
    best_ensemble = results_df['ensemble'].values[0]
    best_reino = float(results_df['self-reflection'].values[0])
    best_term = int(results_df['pred_term'].values[0])
    best_width = int(results_df['width'].values[0])
    best_losscut = float(results_df['losscut'].values[0])
    best_rate = float(results_df['profit_rates'].values[0])
    best_MDD = float(results_df['MDD'].values[0])
    best_product = float(results_df['profit_product'].values[0])

    best_ensemble = best_ensemble.replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')

    import datetime
    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    best_model_path = '../알고리즘트레이딩/best_model.csv'
    df = pd.read_csv(best_model_path, encoding='euc-kr')
    add_model = []
    add_model.append([now, term1, term1,
                     '알4-1_best_model4', top_ensemble, top_reino, top_term, top_width,
                      best_ensemble, best_rate, best_MDD, best_product, 0, best_MDD])

    add_model = pd.DataFrame(np.array(add_model), columns=['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                           'ensemble', 'self-reflection', 'pred_term', 'width',
                                                           'ensemble2',	'profit_rates2', 'MDD2', 'profit_product2',
                                                           'score', 'MDD_sum'])

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