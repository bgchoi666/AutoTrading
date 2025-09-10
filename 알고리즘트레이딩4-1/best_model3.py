# Copyright 2025 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# reinfo, pred_term, losscut은 [0, 0.3, 0.05], [1~5] and [5, 41, 5], [0, 0.01, 0.005]에서 random으로 n번 반복
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

import make_reinfo2 as mr2

ensembles = [

    ['10HL', '15P', '5HL'],
]
random_ensemble = True

cnt = 1000


if __name__ == "__main__":

    ####################################################################################################################
    # 최근 탐색 기간에서 21개 모델 중 3개 선택, reinfo_th, pred_term, width 랜덤으로 선택, 100번 실행
    ####################################################################################################################

    if len(sys.argv) > 1:
        conf.loss_cut = float(sys.argv[1])
    else:
        conf.loss_cut = 0.005

    conf.start_time = '2024/12/16/09:00'
    conf.end_time = '2025/02/15/15:00'
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    conf.gubun = 0

    ep.mr = mr2
    ep.profit.slippage = 0.05
    margin = 10000000
    ep.margin = margin
    ep.profit.margin = margin

    conf.selected_num = 3
    add_txt = "_천만투자_slippage" + str(ep.profit.slippage) + '_reinfo0_random'

    last_trains = [value for value in ep.last_trains if value < "2019-12-31"] + [value for value in ep.last_trains if value > "2020-12-15"]
    start_times = [value for value in ep.start_times if value < "2020/01/01/09:00"] + [value for value in ep.start_times if value > "2020/12/16/09:00"]
    end_times = [value for value in ep.end_times if value < "2020/01/15/15:00"] + [value for value in ep.end_times if value > "2020/12/31/15:00"]
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "eval_reflection_random")

    len_terms1 = len(ep.last_trains)

    #conf.model_pools = ["5C", "5HL", "5P", "20C", "20HL", "20P", "40C", "40HL", "40P"]

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0, 0]
    while len(results) < cnt:

        conf.reinfo_th = 0#random.randrange(0, 31, 5) / 100#random.sample([0.2, 0.3, 0.4, 0.5], 1)[0]
        conf.pred_term = 0#random.sample([1,2,3,4,5,6,10,15,20,25,30,40], 1)[0]#random.randint(1, 40) #random.sample([10, 20, 30, 40], 1)[0]
        conf.reinfo_width = 0#random.randrange(5, 71, 5) #random.sample([10, 20, 30, 40, 50, 60, 70], 1)[0]
        conf.loss_cut = random.randrange(0, 50, 5) / 10000 # random.sample([10, 20, 30, 40, 50, 60, 70], 1)[0]
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

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width: ",
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
    # 기하 수익률이 큰 순으로  20개 모델 선택
    ####################################################################################################################

    length = 20

    # 1. result dataframe profit_product 내림차순 sort 20개 선택
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)[:length]
    top_ensembles = results_df.loc[:, 'ensemble'].values

    ####################################################################################################################
    # 최근 탐색 기간에서 선택된 모델들에 대해 이전 탐색 기간에서 수익률, MDD, 기하 수익률 조사
    ####################################################################################################################

    # 2. 20개 선택 모델로 all_term_ensembl3 실행
    n = conf.selected_num
    for i in range(length):
        top_ensembles[i] = top_ensembles[i].replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
        top_ensembles[i][n] = float(top_ensembles[i][n])
        top_ensembles[i][n+1] = int(top_ensembles[i][n+1])
        top_ensembles[i][n+2] = int(top_ensembles[i][n+2])
        top_ensembles[i][n + 3] = float(top_ensembles[i][n + 3])

    top_ensembles = top_ensembles.tolist()

    import importlib
    importlib.reload(ep)
    import all_term_ensemble3 as a3
    a3.models = top_ensembles
    a3.conf.start_time = '2023/12/16/09:00'
    a3.conf.end_time = '2024/12/15/15:00'

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    a3.ep.start_times[np.where(np.array(a3.ep.start_times) <= a3.conf.start_time)[0][-1]] = a3.conf.start_time
    a3.ep.end_times[np.where(np.array(a3.ep.end_times) >= a3.conf.end_time)[0][0]] = a3.conf.end_time

    # 기간의 달수 계산
    start = datetime.datetime.strptime(a3.conf.start_time, "%Y/%m/%d/09:00")
    end = datetime.datetime.strptime(a3.conf.end_time, "%Y/%m/%d/15:00")
    len_terms2 = (end.year - start.year)*24 + (end.month - start.month + 1)*2

    term2 = a3.conf.start_time[:10].replace('/', '-') + "~" + a3.conf.end_time[:10].replace('/', '-')
    a3.conf.selected_num = 3
    a3.add_txt = '_천만투자_1월31일_reinfo0_random'
    a3.ep.margin = margin
    a3.ep.profit.margin = margin

    eval_path = 'eval_reflection3/eval_reflection3_' + term2 + a3.add_txt + '_losscut0.005.csv'
    if os.path.isfile(eval_path):
        r = pd.read_csv(eval_path, encoding='euc-kr').values
    else:
        r = a3.create_all_term_ensemble3()

    ####################################################################################################################
    # 최근 탐색 기간과 이전 탐색 기간의 기하 수익률, MDD를 고려하여 통합 score 산출, best 모델 선출
    ####################################################################################################################

    # 3. 1, 2의 기하 누적 수익률 / MDD의 합 계산하여 내림차순 sort
    concat_results = np.concatenate((r, results_df.values), axis=1)
    concat_df = pd.DataFrame(concat_results, columns=
            ['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD', 'profit_product',
             'ensemble2', 'self-reflection2', 'pred_term2', 'width2', 'losscut2', 'profit_rates2',
             'MDD2', 'profit_product2']).reset_index(drop=True)
    score = np.zeros(len(concat_results))
    for i in range(len(concat_results)):
        score[i] = pow(float(concat_results[i, 7]), 24 / len_terms2) * \
                   pow(float(concat_results[i, 15]), 24 / len_terms1) / \
                   (float(concat_results[i, 6]) + float(concat_results[i, 14]))
    print(len(ep.last_trains), len(a3.ep.last_trains))
    concat_df['score'] = score
    concat_df = concat_df.sort_values(by=['score'], axis=0, ascending=False).reset_index(drop=True)
    concat_df.to_csv("eval_reflection3/"+"비교_"+term2+"vs"+term1+".csv", encoding='euc-kr')

    ####################################################################################################################
    # 3의 top 5중 최소 MDD합을 가지는 모델 선택
    ####################################################################################################################
    
    MDD_sum = concat_df['MDD'].values.astype(dtype=np.float) + concat_df['MDD2'].values.astype(dtype=np.float)
    concat_df['MDD_sum'] = MDD_sum
    best_df = concat_df[:5].reset_index(drop=True)
    best_idx = best_df['MDD_sum'].idxmin()
    best_model = best_df.loc[best_idx]

    import datetime
    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    best_model_path = '../알고리즘트레이딩/best_model.csv'
    df = pd.read_csv(best_model_path, encoding='euc-kr')
    add_model = []
    add_model.append([now, term1, term2,
                     '알4-1_best_model3', best_model['ensemble'], best_model['self-reflection'], best_model['pred_term'],
                      best_model['width'], best_model['ensemble2'], best_model['profit_rates2'], best_model['MDD2'],
                      best_model['profit_product2'], best_model['score'], best_model['MDD_sum']])

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