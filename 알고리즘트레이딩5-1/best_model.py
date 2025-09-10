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

    ["15HL", "15P", "40C", 0, 10, 100],

]
random_ensemble = True
losscut_mode = ''

#if losscut_mode == 'random':
#    cnt = 200
#else:
cnt = 100


if __name__ == "__main__":

    ####################################################################################################################
    # 최근 탐색 기간에서 21개 모델 중 3개 선택, reinfo_th, pred_term, width 랜덤으로 선택, 100번 실행
    ####################################################################################################################

    if len(sys.argv) > 1:
        loss_cut = float(sys.argv[1])
    else:
        loss_cut = 0.005

    conf.start_time = '2024/08/01/09:00'
    conf.end_time = '2025/07/31/15:00'
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    conf.gubun = 0

    ep.mr = mr2
    mode = 'mr2'
    ep.profit.slippage = 0.05
    margin = 10000000
    ep.margin = margin
    ep.profit.margin = margin
    add_txt = "_천만투자_"+mode+"_slippage" + str(ep.profit.slippage) + losscut_mode

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

                # dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profits_rates}
                # pd.DataFrame(dic).to_excel(path, index = False, encoding = 'euc-kr')

            print("중간 best >>>>")
            print(best, len(results))

            print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width, losscut: ",
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term,
                  conf.reinfo_width, conf.loss_cut)

            results.append([ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut,
                            profit_rates[-1], MDD, profit_product])

            results_df = pd.DataFrame(np.array(results),
                         columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD',
                                  'profit_product'])
            #print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean())

            results_df.to_csv(ep.result_path, index=False)

    print("최종 best >>>> ")
    print(best)

    results_df = pd.DataFrame(np.array(results),
                              columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD',
                                       'profit_product'])

    avg = results_df['profit_rates'].values.astype(dtype=np.float).mean()

    print("평균 >>>> ")
    print(avg)

    #기하 수익률 순으로 정렬, 저장
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)
    results_df.to_csv(ep.result_path, index=False)

    ####################################################################################################################
    # 기하 수익률이 큰 순으로  20개 모델 선택
    ####################################################################################################################

    length = 20

    # 1. result dataframe profit_product 내림차순 sort 20개 선택
    results_df = results_df[:length]
    top_ensembles = results_df.loc[:, 'ensemble'].values

    ####################################################################################################################
    # 최근 탐색 기간에서 선택된 모델들에 대해 이전 탐색 기간에서 수익률, MDD, 기하 수익률 조사
    ####################################################################################################################

    # 2. 20개 선택 모델로 all_term_ensembl3 실행
    for i in range(length):
        top_ensembles[i] = top_ensembles[i].replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
        top_ensembles[i][3] = float(top_ensembles[i][3])
        top_ensembles[i][4] = int(top_ensembles[i][4])
        top_ensembles[i][5] = int(top_ensembles[i][5])
        top_ensembles[i][6] = float(top_ensembles[i][6])


    top_ensembles = top_ensembles.tolist()

    import importlib
    importlib.reload(ep)
    import all_term_ensemble3 as a3
    a3.models = top_ensembles
    a3.conf.start_time = '2022/08/01/09:00'
    a3.conf.end_time = '2024/07/31/15:00'
    a3.ep.margin = margin
    a3.ep.profit.margin = margin
    a3.ep.mr = mr3

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    a3.ep.start_times[np.where(np.array(a3.ep.start_times) <= a3.conf.start_time)[0][-1]] = a3.conf.start_time
    a3.ep.end_times[np.where(np.array(a3.ep.end_times) >= a3.conf.end_time)[0][0]] = a3.conf.end_time

    # 기간의 달수 계산
    start = datetime.datetime.strptime(a3.conf.start_time, "%Y/%m/%d/09:00")
    end = datetime.datetime.strptime(a3.conf.end_time, "%Y/%m/%d/15:00")
    len_terms2 = (end.year - start.year)*24 + (end.month - start.month + 1)*2

    term2 = a3.conf.start_time[:10].replace('/', '-') + "~" + a3.conf.end_time[:10].replace('/', '-')
    a3.add_txt = '_천만투자_'+mode+'_7월31일' + losscut_mode
    a3.ep.conf.loss_cut = loss_cut

    eval_path = 'eval_reflection3/eval_reflection3_' + term2 + a3.add_txt + '_losscut' + str(loss_cut) + '.csv'
    #if os.path.isfile(eval_path):
    #    r = pd.read_csv(eval_path, encoding='euc-kr').values
    #else:
    r = a3.create_all_term_ensemble3()

    ####################################################################################################################
    # 최근 탐색 기간과 이전 탐색 기간의 기하 수익률, MDD를 고려하여 통합 score 산출, best 모델 선출
    ####################################################################################################################

    # 3. 1, 2의 기하 누적 수익률 / MDD의 합 계산하여 내림차순 sort
    concat_results = np.concatenate((r, results_df.values), axis=1)
    concat_df = pd.DataFrame(concat_results, columns=
            ['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD', 'profit_product',
             'ensemble2', 'self-reflection2', 'pred_term2', 'width2', 'losscut2', 'profit_rates2', 'MDD2', 'profit_product2']).reset_index(drop=True)
    score = np.zeros(len(concat_results))
    for i in range(len(concat_results)):
        score[i] = pow(float(concat_results[i, 7]), 24 / len_terms2) * \
                   pow(float(concat_results[i, 15]), 24 / len_terms1) / \
                   (float(concat_results[i, 6]) + float(concat_results[i, 14]))
    concat_df['score'] = score
    concat_df = concat_df.sort_values(by=['score'], axis=0, ascending=False).reset_index(drop=True)
    concat_df.to_csv("eval_reflection3/"+"비교_m3_"+term2+"vs"+term1+"_"+losscut_mode+"losscut"+str(conf.loss_cut)+".csv", index=False, encoding='euc-kr')

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
                     '알5-1_'+losscut_mode+'1년_2년_'+mode, best_model['ensemble'], best_model['self-reflection'], best_model['pred_term'], best_model['width'],
                     best_model['ensemble2'], best_model['profit_rates2'], best_model['MDD2'], best_model['profit_product2'],
                     best_model['score'], best_model['MDD_sum']])

    add_model = pd.DataFrame(np.array(add_model), columns=['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                           'ensemble', 'self-reflection', 'pred_term', 'width',
                                                           'ensemble2',	'profit_rates2', 'MDD2', 'profit_product2',
                                                           'score', 'MDD_sum'])
    if losscut_mode == 'random':
        df = pd.concat([df, add_model], axis=0)[['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                 'ensemble', 'self-reflection', 'pred_term', 'width',
                                                 'ensemble2', 'profit_rates2', 'MDD2', 'profit_product2',
                                                 'score', 'MDD_sum']]

        df.to_csv(best_model_path, encoding='euc-kr', index=False)

        exit(0)

    ####################################################################################################################
    # bayesian_losscut으로 최근 15일 ~ 1개월 기간에서 최종 학습된 모델의 최적화된 losscut 탐색, 첨부하여 저장
    ####################################################################################################################

    import baysian_losscut as bl

    org_model = list(add_model.loc[0, 'ensemble2'])
    org_model = org_model[:6]

    bl.ep_proc = False
    bl.ep.margin = margin
    bl.ep.profit.margin = margin
    ensembles = [

        org_model

    ]
    start_time = '2025/06/01/09:00'
    end_time = '2025/07/31/15:00'
    last_train = '2025-07-31'

    bl.low = 0.001
    bl.high = 0.01

    try:
        losscut_added_model, _, _ = bl.create_losscut(ensembles, start_time, end_time, last_train)
    except:
        org_model.append(0.005)
        losscut_added_model = org_model
    print("best_losscut: ", str(losscut_added_model))

    # losscut을 첨부하여 모델 저장
    add_model.loc[0, 'ensemble2'] = losscut_added_model

    df = pd.concat([df, add_model], axis=0)[['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                           'ensemble', 'self-reflection', 'pred_term', 'width',
                                                           'ensemble2',	'profit_rates2', 'MDD2', 'profit_product2',
                                                           'score', 'MDD_sum']]

    df.to_csv(best_model_path, encoding='euc-kr', index=False)

    exit(0)

    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, close_rates, color='r', linewidth=3.0, label='kospi200 f index')
    ensemble_name = 'ensemble ' + ensemble_models
    plt.plot(dates, profit_rates, color='b', linewidth=3.0, label=ensemble_name)
    plt.legend()

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))

    plt.xticks(rotation=30)

    plt.show()
    """