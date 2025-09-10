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

import make_reinfo3 as mr3
import make_reinfo2 as mr2
import make_reinfo as mr

ensembles = [

    ["30HL", "30P", "40P"],
]
random_ensemble = True
losscut_mode = 'random'

cnt = 1000

if len(sys.argv) > 1:
    loss_cut = float(sys.argv[1])
else:
    loss_cut = 0.005

if __name__ == "__main__":

    ####################################################################################################################
    # random 앙상블 (21개 모델 중 3개 선택), reinfo_th, pred_term, width 랜덤으로 선택, 1000번 실행
    ####################################################################################################################

    random_ensemble = True
    losscut_mode = 'random'

    cnt = 1000

    conf.start_time = '2025/04/01/09:00'
    conf.end_time = '2025/06/30/15:00'
    term1 = conf.start_time[:10].replace('/', '-') + "~" + conf.end_time[:10].replace('/', '-')

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    conf.gubun = 0
    # 앙상블내의 모델 개수 설정
    conf.selected_num = 3

    ep.mr = mr3
    mode = 'mr3'
    ep.profit.slippage = 0.05
    ep.margin = 10000000
    ep.profit.margin = 10000000
    #ep.profit.trading_9h = True
    if ep.profit.trading_9h == True:
        add_txt = "_천만투자_trading_9h_slippage" + str(ep.profit.slippage) + losscut_mode
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

    if len(results) >= cnt:
        results_df = pd.DataFrame(np.array(results),
                                  columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                           'profit_rates', 'MDD', 'profit_product'])

    while len(results) < cnt:

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

        conf.selected_model_types = sorted(random.sample(conf.model_pools, conf.selected_num))

        ensemble_models = '["'
        for i in range(conf.selected_num - 1):
            ensemble_models = ensemble_models + conf.selected_model_types[i] + '", "'
        ensemble_models = ensemble_models + conf.selected_model_types[2] + '", ' + str(conf.reinfo_th) + ', ' + str(
            conf.pred_term) + ', ' + str(conf.reinfo_width) + ', ' + str(conf.loss_cut) + ']'
        if not results == [] and ensemble_models in np.array(results)[:, 0].tolist():
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

            # dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
            # pd.DataFrame(dic).to_excel(path, index=False)

        print("중간 best >>>>")
        print(best)

        print("profit rates, MDD, profit_product, ensemble, reinfo, pred_term, width, losscut: ",
              profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term,
              conf.reinfo_width, conf.loss_cut)
        results.append(
            [ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut, profit_rates[-1], MDD,
             profit_product])
        results_df = pd.DataFrame(np.array(results),
                                  columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                           'profit_rates', 'MDD', 'profit_product'])

        print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean(), len(results))
        results_df.to_csv(ep.result_path, index=False)

    # profit_product 순으로 재배열해서 저장
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)
    results_df.to_csv(ep.result_path, index=False)
    print('best: ', results_df['ensemble'].values[0], results_df['profit_product'].values[0])

    ####################################################################################################################
    # product_rate순으로 상위 5개 앙상블(3개 모델 조합) 선택 후 reinfo_th, pred_term, width 랜덤으로 선택, 100번 실행
    ####################################################################################################################

    # result dataframe profit_product 내림차순 sort 5개 선택
    results_df = results_df[:5]
    top_ensembles = results_df.loc[:, 'ensemble'].values

    n = conf.selected_num
    for i in range(5):
        top_ensembles[i] = top_ensembles[i].replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
        top_ensembles[i] = top_ensembles[i][:n]

    ensembles = top_ensembles

    # 선택한 5개 앙상블에 대해 random 변수들에 대한 수익률 조사
    random_ensemble = False
    losscut_mode = 'random'

    cnt = 100

    # path 지정
    add_txt = "_앙상블5개고정_천만투자_" + mode + "_slippage" + str(ep.profit.slippage) + losscut_mode
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, last_trains, start_times, end_times, add_txt)
    ep.folder, ep.result_path = ep.set_path("앙상블실험_random", "앙상블고정_eval_reflection_random")

    if not os.path.isfile(ep.result_path):
        results = []
    else:
        results = pd.read_csv(ep.result_path).values.tolist()
    best = [0, 0, 0, '', 0, 0, 0, 0]

    if len(results) >= cnt:
        results_df = pd.DataFrame(np.array(results),
                                  columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut',
                                           'profit_rates', 'MDD',
                                           'profit_product'])

    while len(results) < cnt:

        for enm in ensembles:

            conf.reinfo_th = random.randrange(0, 101, 5) / 100

            if mode != 'mr3':
                conf.pred_term = random.sample([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40], 1)[0]
                if conf.reinfo_th > 0.5 and conf.pred_term > 5:
                    continue

            conf.reinfo_width = random.randrange(5, 71, 5)

            if losscut_mode == 'random':
                conf.loss_cut = random.randrange(10, 51, 5) / 10000
            else:
                conf.loss_cut = loss_cut

            conf.target_type = 'C'

            conf.selected_model_types = enm

            ensemble_models = '["'
            for i in range(conf.selected_num - 1):
                ensemble_models = ensemble_models + conf.selected_model_types[i] + '", "'
            ensemble_models = ensemble_models + conf.selected_model_types[2] + '", ' + str(conf.reinfo_th) + ', ' + str(
                conf.pred_term) + ', ' + str(conf.reinfo_width) + ', ' + str(conf.loss_cut) + ']'
            if not results == [] and ensemble_models in np.array(results)[:, 0].tolist():
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
                  profit_rates[-1], MDD, profit_product, conf.selected_model_types, conf.reinfo_th, conf.pred_term,
                  conf.reinfo_width, conf.loss_cut)
            results.append(
                [ensemble_models, conf.reinfo_th, conf.pred_term, conf.reinfo_width, conf.loss_cut, profit_rates[-1], MDD,
                 profit_product])
            results_df = pd.DataFrame(np.array(results),
                                      columns=['ensemble', 'self-reflection', 'pred_term', 'width', 'losscut', 'profit_rates',
                                               'MDD', 'profit_product'])

            print('중간 평균: ', results_df['profit_rates'].values.astype(dtype=np.float).mean(), len(results))
            results_df.to_csv(ep.result_path, index=False)

    # profit_product 순으로 재배열해서 저장
    results_df = results_df.sort_values(by=['profit_product'], axis=0, ascending=False).reset_index(drop=True)
    results_df.to_csv(ep.result_path, index=False)
    best_model = results_df.loc[0]
    print('best: ', best_model, results_df['profit_product'].values[0])

    # best_model 저장
    import datetime
    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")

    best_model_path = '../알고리즘트레이딩/best_model.csv'
    df = pd.read_csv(best_model_path, encoding='euc-kr')
    add_model = []
    add_model.append([now, term1, term1,
                     '알2-1_new_'+losscut_mode+'3개월', best_model['ensemble'], best_model['self-reflection'], best_model['pred_term'], best_model['width'],
                      best_model['ensemble'], best_model['profit_rates'], best_model['MDD'], best_model['profit_product'], 0, 0])

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