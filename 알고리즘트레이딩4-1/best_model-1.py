# Copyright 2025 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 기간별 추천 앙상블의 수익률 생성
# reinfo, pred_term은 [0, 1], [1, 40], (width는 100으로 고정) losscut은 [0.001, 0.,005]에서 random으로 1000번 반복
# 기하 수익률 best 100을 뽑아 그 이전 기간(처음 기간의 2배)에서 다시 수익률 조사
# 기하 수익률 best 20를 뽑아 그 이전의 이전 기간 (이전 기간의 2배)에서 다시 수익률 조사
# 기하 수익룰 best를 선택
# 예) 2024-10-01~2024-12-31 에서 random으로 1000번 수익률 조사, best 100모델에 대해 2024-04-01~2024-09-30에서 수익률 조사
#     다시 best 20에 대해 2023-04-01~2024-03-31에서 수익률 조사하여 기하 수이률 best인 모델 선택

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
losscut_mode = 'random'

cnt = 1000


if __name__ == "__main__":

    ####################################################################################################################
    # 최근 탐색 기간에서 21개 모델 중 3개 선택, reinfo_th, pred_term, width 랜덤으로 선택, 100번 실행
    ####################################################################################################################

    if len(sys.argv) > 1:
        loss_cut = float(sys.argv[1])
    else:
        loss_cut = 0.005

    conf.start_time = '2025/06/01/09:00'
    conf.end_time = '2025/08/31/15:00'
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    conf.gubun = 0

    ep.mr = mr3
    mode = 'mr3'
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
    # 기하 수익률이 큰 순으로  100개 모델 선택
    ####################################################################################################################

    length = 100

    # 1. result dataframe profit_product 내림차순 sort 20개 선택
    results_df = results_df[:length]
    top_ensembles = results_df.loc[:, 'ensemble'].values

    ####################################################################################################################
    # 최근 탐색 기간에서 선택된 모델들에 대해 이전 탐색 기간에서 수익률, MDD, 기하 수익률 조사
    ####################################################################################################################

    # 2. 100개 선택 모델로 all_term_ensembl3 실행
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
    a3.conf.start_time = '2024/12/01/09:00'
    a3.conf.end_time = '2025/05/31/15:00'
    a3.ep.margin = margin
    a3.ep.profit.margin = margin
    a3.ep.mr = ep.mr

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    a3.ep.start_times[np.where(np.array(a3.ep.start_times) <= a3.conf.start_time)[0][-1]] = a3.conf.start_time
    a3.ep.end_times[np.where(np.array(a3.ep.end_times) >= a3.conf.end_time)[0][0]] = a3.conf.end_time

    # 기간의 달수 계산
    start = datetime.datetime.strptime(a3.conf.start_time, "%Y/%m/%d/09:00")
    end = datetime.datetime.strptime(a3.conf.end_time, "%Y/%m/%d/15:00")
    len_terms2 = (end.year - start.year)*24 + (end.month - start.month + 1)*2

    term2 = a3.conf.start_time[:10].replace('/', '-') + "~" + a3.conf.end_time[:10].replace('/', '-')
    a3.add_txt = '이전기간_'+mode+'_천만투자_8월31일' + losscut_mode
    a3.ep.conf.loss_cut = loss_cut

    eval_path = 'eval_reflection3/eval_reflection3_' + mode + '_' + term2 + a3.add_txt + '_losscut' + str(loss_cut) + '.csv'
    #if os.path.isfile(eval_path):
    #    r = pd.read_csv(eval_path, encoding='euc-kr').values
    #else:
    r = a3.create_all_term_ensemble3()

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
    concat_df.to_csv("eval_reflection3/"+"비교_"+mode+"_이전_최근_"+term2+"vs"+term1+"_"+losscut_mode+"losscut"+str(conf.loss_cut)+".csv", index=False, encoding='euc-kr')

    ####################################################################################################################
    # 이전 기간의 top 20를 선택하여 이전 기간의 2배 길이의 처음 기간에서 수익률 조사
    ####################################################################################################################

    length = 20

    best_df = concat_df[:20].reset_index(drop=True)
    best_df = best_df.drop('score', axis=1)

    top_ensembles = best_df.loc[:, 'ensemble2'].values

    # 20개 선택 모델로 all_term_ensembl3 실행
    #for i in range(length):
    #    top_ensembles[i] = top_ensembles[i].replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
    #    top_ensembles[i][3] = float(top_ensembles[i][3])
    #    top_ensembles[i][4] = int(top_ensembles[i][4])
    #    top_ensembles[i][5] = int(top_ensembles[i][5])
    #    top_ensembles[i][6] = float(top_ensembles[i][6])

    import importlib
    importlib.reload(ep)
    import all_term_ensemble3 as a3
    a3.models = top_ensembles.tolist()
    a3.conf.start_time = '2023/12/01/09:00'
    a3.conf.end_time = '2024/11/30/15:00'
    a3.ep.margin = margin
    a3.ep.profit.margin = margin
    a3.ep.mr = ep.mr

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    a3.ep.start_times[np.where(np.array(a3.ep.start_times) <= a3.conf.start_time)[0][-1]] = a3.conf.start_time
    a3.ep.end_times[np.where(np.array(a3.ep.end_times) >= a3.conf.end_time)[0][0]] = a3.conf.end_time

    # 기간의 달수 계산
    start = datetime.datetime.strptime(a3.conf.start_time, "%Y/%m/%d/09:00")
    end = datetime.datetime.strptime(a3.conf.end_time, "%Y/%m/%d/15:00")
    len_terms3 = (end.year - start.year)*24 + (end.month - start.month + 1)*2

    term3 = a3.conf.start_time[:10].replace('/', '-') + "~" + a3.conf.end_time[:10].replace('/', '-')
    a3.add_txt = '처음기간_'+mode+'_천만투자_8월31일' + losscut_mode
    a3.ep.conf.loss_cut = loss_cut

    eval_path = 'eval_reflection3/eval_reflection3_' + mode + '_' + term3 + a3.add_txt + '_losscut' + str(loss_cut) + '.csv'
    #if os.path.isfile(eval_path):
    #    r = pd.read_csv(eval_path, encoding='euc-kr').values
    #else:
    r2 = a3.create_all_term_ensemble3()

    # 처음, 이전 기간의 기하 누적 수익률 / MDD의 합 계산하여 내림차순 sort
    concat_results = np.concatenate((r2, best_df.values), axis=1)
    concat_df = pd.DataFrame(concat_results, columns=
            ['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates', 'MDD', 'profit_product',
             'ensemble1', 'self-reflection1', 'pred_term1', 'width1', 'losscut1', 'profit_rates1', 'MDD1', 'profit_product1',
             'ensemble2', 'self-reflection2', 'pred_term2', 'width2', 'losscut2', 'profit_rates2', 'MDD2', 'profit_product2']).reset_index(drop=True)
    score = np.zeros(len(concat_results))
    for i in range(len(concat_results)):
        score[i] = pow(float(concat_results[i, 7]), 24 / len_terms3) * \
                   pow(float(concat_results[i, 15]), 24 / len_terms2) * \
                   pow(float(concat_results[i, 23]), 24 / len_terms1) / \
                   (float(concat_results[i, 6]) + float(concat_results[i, 14]) + float(concat_results[i, 22]))
    concat_df['score'] = score
    concat_df = concat_df.sort_values(by=['score'], axis=0, ascending=False).reset_index(drop=True)
    concat_df.to_csv("eval_reflection3/"+"비교_"+mode+"_처음_이전_최근_"+term3+"vs"+term2+"_"+term1+"_"+losscut_mode+"losscut"+str(conf.loss_cut)+".csv", index=False, encoding='euc-kr')

    MDD_sum = concat_df['MDD'].values.astype(dtype=np.float) + concat_df['MDD1'].values.astype(dtype=np.float) + concat_df['MDD2'].values.astype(dtype=np.float)
    concat_df['MDD_sum'] = MDD_sum
    best_df = concat_df[:5].reset_index(drop=True)
    best_idx = best_df['MDD_sum'].idxmin()
    best_model = best_df.loc[best_idx]

    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    best_model_path = '../알고리즘트레이딩/best_model.csv'
    df = pd.read_csv(best_model_path, encoding='euc-kr')
    add_model = []
    add_model.append([now, term1, term2,
                     '알4-1_'+losscut_mode+'3개월_6개월_12개월_'+mode, best_model['ensemble'], best_model['self-reflection'], best_model['pred_term'], best_model['width'],
                     best_model['ensemble2'], best_model['profit_rates2'], best_model['MDD2'], best_model['profit_product2'],
                     best_model['score'], best_model['MDD_sum']])

    add_model = pd.DataFrame(np.array(add_model), columns=['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                                           'ensemble', 'self-reflection', 'pred_term', 'width',
                                                           'ensemble2',	'profit_rates2', 'MDD2', 'profit_product2',
                                                           'score', 'MDD_sum'])

    df = pd.concat([df, add_model], axis=0)[['date', '최근탐색기간', '이전탐색기간', '알고리즘',
                                             'ensemble', 'self-reflection', 'pred_term', 'width',
                                             'ensemble2', 'profit_rates2', 'MDD2', 'profit_product2',
                                             'score', 'MDD_sum']]


    df.to_csv(best_model_path, encoding='euc-kr', index=False)

    exit(0)

