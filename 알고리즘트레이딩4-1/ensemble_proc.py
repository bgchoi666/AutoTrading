# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 앙상블의 기간별 수익률 조사와 관련된 모듈들

import data
from data import config as conf
import model as md
import make_reinfo as mr
import profit
import baysian as bs

import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random

# warning message 안나오게. . .
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

last_trains = [
    '2016-12-31',
    '2017-01-15', '2017-01-31', '2017-02-15', '2017-02-28',
    '2017-03-15', '2017-03-31', '2017-04-15', '2017-04-30',
    '2017-05-15', '2017-05-31', '2017-06-15', '2017-06-30',
    '2017-07-15', '2017-07-31', '2017-08-15', '2017-08-31',
    '2017-09-15', '2017-09-30', '2017-10-15', '2017-10-31',
    '2017-11-15', '2017-11-30', '2017-12-15', '2017-12-31',

    '2018-01-15', '2018-01-31', '2018-02-15', '2018-02-28',
    '2018-03-15', '2018-03-31', '2018-04-15', '2018-04-30',
    '2018-05-15', '2018-05-31', '2018-06-15', '2018-06-30',
    '2018-07-15', '2018-07-31', '2018-08-15', '2018-08-31',
    '2018-09-15', '2018-09-30', '2018-10-15', '2018-10-31',
    '2018-11-15', '2018-11-30', '2018-12-15',

    '2018-12-31',
    '2019-01-15', '2019-01-31', '2019-02-15', '2019-02-28',
    '2019-03-15', '2019-03-31', '2019-04-15', '2019-04-30',
    '2019-05-15', '2019-05-31', '2019-06-15', '2019-06-30',
    '2019-07-15', '2019-07-31', '2019-08-15', '2019-08-31',
    '2019-09-15', '2019-09-30', '2019-10-15', '2019-10-31',
    '2019-11-15', '2019-11-30', '2019-12-15',

    '2019-12-31',
    '2020-01-15', '2020-01-31', '2020-02-15', '2020-02-28',
    '2020-03-15', '2020-03-31', '2020-04-15', '2020-04-30',
    '2020-05-15', '2020-05-31', '2020-06-15', '2020-06-30',
    '2020-07-15', '2020-07-31', '2020-08-15', '2020-08-31',
    '2020-09-15', '2020-09-30', '2020-10-15', '2020-10-31',
    '2020-11-15', '2020-11-30', '2020-12-15',

    '2020-12-31',
    '2021-01-15', '2021-01-31', '2021-02-15', '2021-02-28',
    '2021-03-15', '2021-03-31', '2021-04-15', '2021-04-30',
    '2021-05-15', '2021-05-31', '2021-06-15', '2021-06-30',
    '2021-07-15', '2021-07-31', '2021-08-15', '2021-08-31',
    '2021-09-15', '2021-09-30', '2021-10-15', '2021-10-31',
    '2021-11-15', '2021-11-30', '2021-12-15',

    '2021-12-31',
    '2022-01-15', '2022-01-31', '2022-02-15', '2022-02-28',
    '2022-03-15', '2022-03-31', '2022-04-15', '2022-04-30',
    '2022-05-15', '2022-05-31', '2022-06-15', '2022-06-30',
    '2022-07-15', '2022-07-31', '2022-08-15', '2022-08-31',
    '2022-09-15', '2022-09-30', '2022-10-15', '2022-10-31',
    '2022-11-15', '2022-11-30', '2022-12-15',

    '2022-12-31',
    '2023-01-15', '2023-01-31', '2023-02-15', '2023-02-28',
    '2023-03-15', '2023-03-31', '2023-04-15', '2023-04-30',
    '2023-05-15', '2023-05-31', '2023-06-15', '2023-06-30',
    '2023-07-15', '2023-07-31', '2023-08-15', '2023-08-31',
    '2023-09-15', '2023-09-30', '2023-10-15', '2023-10-31',
    '2023-11-15', '2023-11-30', '2023-12-15',

    '2023-12-31',
    '2024-01-15', '2024-01-31', '2024-02-15', '2024-02-29',
    '2024-03-15', '2024-03-31', '2024-04-15', '2024-04-30',
    '2024-05-15', '2024-05-31', '2024-06-15', '2024-06-30',
    '2024-07-15', '2024-07-31', '2024-08-15', '2024-08-31',
    '2024-09-15', '2024-09-30', '2024-10-15', '2024-10-31',
    '2024-11-15', '2024-11-30', '2024-12-15',

    '2024-12-31',
    '2025-01-15', '2025-01-31', '2025-02-15', '2025-02-28',
    '2025-03-15', '2025-03-31', '2025-04-15', '2025-04-30',
    '2025-05-15', '2025-05-31', '2025-06-15', '2025-06-30',
    '2025-07-15', '2025-07-31', '2025-08-15', '2025-08-31',
    '2025-09-15', '2025-09-30', '2025-10-15', '2025-10-31',
    '2025-11-15', '2025-11-30', '2025-12-15',
]

start_times = [
    '2017/01/01/09:00', '2017/01/16/09:00', '2017/02/01/09:00', '2017/02/16/09:00',
    '2017/03/01/09:00', '2017/03/16/09:00', '2017/04/01/09:00', '2017/04/16/09:00',
    '2017/05/01/09:00', '2017/05/16/09:00', '2017/06/01/09:00', '2017/06/16/09:00',
    '2017/07/01/09:00', '2017/07/16/09:00', '2017/08/01/09:00', '2017/08/16/09:00',
    '2017/09/01/09:00', '2017/09/16/09:00', '2017/10/01/09:00', '2017/10/16/09:00',
    '2017/11/01/09:00', '2017/11/16/09:00', '2017/12/01/09:00', '2017/12/16/09:00',

    '2018/01/01/09:00', '2018/01/16/09:00', '2018/02/01/09:00', '2018/02/16/09:00',
    '2018/03/01/09:00', '2018/03/16/09:00', '2018/04/01/09:00', '2018/04/16/09:00',
    '2018/05/01/09:00', '2018/05/16/09:00', '2018/06/01/09:00', '2018/06/16/09:00',
    '2018/07/01/09:00', '2018/07/16/09:00', '2018/08/01/09:00', '2018/08/16/09:00',
    '2018/09/01/09:00', '2018/09/16/09:00', '2018/10/01/09:00', '2018/10/16/09:00',
    '2018/11/01/09:00', '2018/11/16/09:00', '2018/12/01/09:00', '2018/12/16/09:00',

    '2019/01/01/09:00', '2019/01/16/09:00', '2019/02/01/09:00', '2019/02/16/09:00',
    '2019/03/01/09:00', '2019/03/16/09:00', '2019/04/01/09:00', '2019/04/16/09:00',
    '2019/05/01/09:00', '2019/05/16/09:00', '2019/06/01/09:00', '2019/06/16/09:00',
    '2019/07/01/09:00', '2019/07/16/09:00', '2019/08/01/09:00', '2019/08/16/09:00',
    '2019/09/01/09:00', '2019/09/16/09:00', '2019/10/01/09:00', '2019/10/16/09:00',
    '2019/11/01/09:00', '2019/11/16/09:00', '2019/12/01/09:00', '2019/12/16/09:00',

    '2020/01/01/09:00', '2020/01/16/09:00', '2020/02/01/09:00', '2020/02/16/09:00',
    '2020/03/01/09:00', '2020/03/16/09:00', '2020/04/01/09:00', '2020/04/16/09:00',
    '2020/05/01/09:00', '2020/05/16/09:00', '2020/06/01/09:00', '2020/06/16/09:00',
    '2020/07/01/09:00', '2020/07/16/09:00', '2020/08/01/09:00', '2020/08/16/09:00',
    '2020/09/01/09:00', '2020/09/16/09:00', '2020/10/01/09:00', '2020/10/16/09:00',
    '2020/11/01/09:00', '2020/11/16/09:00', '2020/12/01/09:00', '2020/12/16/09:00',

    '2021/01/01/09:00', '2021/01/16/09:00', '2021/02/01/09:00', '2021/02/16/09:00',
    '2021/03/01/09:00', '2021/03/16/09:00', '2021/04/01/09:00', '2021/04/16/09:00',
    '2021/05/01/09:00', '2021/05/16/09:00', '2021/06/01/09:00', '2021/06/16/09:00',
    '2021/07/01/09:00', '2021/07/16/09:00', '2021/08/01/09:00', '2021/08/16/09:00',
    '2021/09/01/09:00', '2021/09/16/09:00', '2021/10/01/09:00', '2021/10/16/09:00',
    '2021/11/01/09:00', '2021/11/16/09:00', '2021/12/01/09:00', '2021/12/16/09:00',

    '2022/01/01/09:00', '2022/01/16/09:00', '2022/02/01/09:00', '2022/02/16/09:00',
    '2022/03/01/09:00', '2022/03/16/09:00', '2022/04/01/09:00', '2022/04/16/09:00',
    '2022/05/01/09:00', '2022/05/16/09:00', '2022/06/01/09:00', '2022/06/16/09:00',
    '2022/07/01/09:00', '2022/07/16/09:00', '2022/08/01/09:00', '2022/08/16/09:00',
    '2022/09/01/09:00', '2022/09/16/09:00', '2022/10/01/09:00', '2022/10/16/09:00',
    '2022/11/01/09:00', '2022/11/16/09:00', '2022/12/01/09:00', '2022/12/16/09:00',

    '2023/01/01/09:00', '2023/01/16/09:00', '2023/02/01/09:00', '2023/02/16/09:00',
    '2023/03/01/09:00', '2023/03/16/09:00', '2023/04/01/09:00', '2023/04/16/09:00',
    '2023/05/01/09:00', '2023/05/16/09:00', '2023/06/01/09:00', '2023/06/16/09:00',
    '2023/07/01/09:00', '2023/07/16/09:00', '2023/08/01/09:00', '2023/08/16/09:00',
    '2023/09/01/09:00', '2023/09/16/09:00', '2023/10/01/09:00', '2023/10/16/09:00',
    '2023/11/01/09:00', '2023/11/16/09:00', '2023/12/01/09:00', '2023/12/16/09:00',

    '2024/01/01/09:00', '2024/01/16/09:00', '2024/02/01/09:00', '2024/02/16/09:00',
    '2024/03/01/09:00', '2024/03/16/09:00', '2024/04/01/09:00', '2024/04/16/09:00',
    '2024/05/01/09:00', '2024/05/16/09:00', '2024/06/01/09:00', '2024/06/16/09:00',
    '2024/07/01/09:00', '2024/07/16/09:00', '2024/08/01/09:00', '2024/08/16/09:00',
    '2024/09/01/09:00', '2024/09/16/09:00', '2024/10/01/09:00', '2024/10/16/09:00',
    '2024/11/01/09:00', '2024/11/16/09:00', '2024/12/01/09:00', '2024/12/16/09:00',

    '2025/01/01/09:00', '2025/01/16/09:00', '2025/02/01/09:00', '2025/02/16/09:00',
    '2025/03/01/09:00', '2025/03/16/09:00', '2025/04/01/09:00', '2025/04/16/09:00',
    '2025/05/01/09:00', '2025/05/16/09:00', '2025/06/01/09:00', '2025/06/16/09:00',
    '2025/07/01/09:00', '2025/07/16/09:00', '2025/08/01/09:00', '2025/08/16/09:00',
    '2025/09/01/09:00', '2025/09/16/09:00', '2025/10/01/09:00', '2025/10/16/09:00',
    '2025/11/01/09:00', '2025/11/16/09:00', '2025/12/01/09:00', '2025/12/16/09:00',
]

end_times = [
    '2017/01/15/15:00', '2017/01/31/15:00', '2017/02/15/15:00', '2017/02/28/15:00',
    '2017/03/15/15:00', '2017/03/31/15:00', '2017/04/15/15:00', '2017/04/30/15:00',
    '2017/05/15/15:00', '2017/05/31/15:00', '2017/06/15/15:00', '2017/06/30/15:00',
    '2017/07/15/15:00', '2017/07/31/15:00', '2017/08/15/15:00', '2017/08/31/15:00',
    '2017/09/15/15:00', '2017/09/30/15:00', '2017/10/15/15:00', '2017/10/31/15:00',
    '2017/11/15/15:00', '2017/11/30/15:00', '2017/12/15/15:00', '2017/12/31/15:00',

    '2018/01/15/15:00', '2018/01/31/15:00', '2018/02/15/15:00', '2018/02/28/15:00',
    '2018/03/15/15:00', '2018/03/31/15:00', '2018/04/15/15:00', '2018/04/30/15:00',
    '2018/05/15/15:00', '2018/05/31/15:00', '2018/06/15/15:00', '2018/06/30/15:00',
    '2018/07/15/15:00', '2018/07/31/15:00', '2018/08/15/15:00', '2018/08/31/15:00',
    '2018/09/15/15:00', '2018/09/30/15:00', '2018/10/15/15:00', '2018/10/31/15:00',
    '2018/11/15/15:00', '2018/11/30/15:00', '2018/12/15/15:00', '2018/12/31/15:00',

    '2019/01/15/15:00', '2019/01/31/15:00', '2019/02/15/15:00', '2019/02/28/15:00',
    '2019/03/15/15:00', '2019/03/31/15:00', '2019/04/15/15:00', '2019/04/30/15:00',
    '2019/05/15/15:00', '2019/05/31/15:00', '2019/06/15/15:00', '2019/06/30/15:00',
    '2019/07/15/15:00', '2019/07/31/15:00', '2019/08/15/15:00', '2019/08/31/15:00',
    '2019/09/15/15:00', '2019/09/30/15:00', '2019/10/15/15:00', '2019/10/31/15:00',
    '2019/11/15/15:00', '2019/11/30/15:00', '2019/12/15/15:00', '2019/12/31/15:00',

    '2020/01/15/15:00', '2020/01/31/15:00', '2020/02/15/15:00', '2020/02/28/15:00',
    '2020/03/15/15:00', '2020/03/31/15:00', '2020/04/15/15:00', '2020/04/30/15:00',
    '2020/05/15/15:00', '2020/05/31/15:00', '2020/06/15/15:00', '2020/06/30/15:00',
    '2020/07/15/15:00', '2020/07/31/15:00', '2020/08/15/15:00', '2020/08/31/15:00',
    '2020/09/15/15:00', '2020/09/30/15:00', '2020/10/15/15:00', '2020/10/31/15:00',
    '2020/11/15/15:00', '2020/11/30/15:00', '2020/12/15/15:00', '2020/12/31/15:00',

    '2021/01/15/15:00', '2021/01/31/15:00', '2021/02/15/15:00', '2021/02/28/15:00',
    '2021/03/15/15:00', '2021/03/31/15:00', '2021/04/15/15:00', '2021/04/30/15:00',
    '2021/05/15/15:00', '2021/05/31/15:00', '2021/06/15/15:00', '2021/06/30/15:00',
    '2021/07/15/15:00', '2021/07/31/15:00', '2021/08/15/15:00', '2021/08/31/15:00',
    '2021/09/15/15:00', '2021/09/30/15:00', '2021/10/15/15:00', '2021/10/31/15:00',
    '2021/11/15/15:00', '2021/11/30/15:00', '2021/12/15/15:00', '2021/12/31/15:00',

    '2022/01/15/15:00', '2022/01/31/15:00', '2022/02/15/15:00', '2022/02/28/15:00',
    '2022/03/15/15:00', '2022/03/31/15:00', '2022/04/15/15:00', '2022/04/30/15:00',
    '2022/05/15/15:00', '2022/05/31/15:00', '2022/06/15/15:00', '2022/06/30/15:00',
    '2022/07/15/15:00', '2022/07/31/15:00', '2022/08/15/15:00', '2022/08/31/15:00',
    '2022/09/15/15:00', '2022/09/30/15:00', '2022/10/15/15:00', '2022/10/31/15:00',
    '2022/11/15/15:00', '2022/11/30:15:00', '2022/12/15/15:00', '2022/12/31/15:00',

    '2023/01/15/15:00', '2023/01/31/15:00', '2023/02/15/15:00', '2023/02/28/15:00',
    '2023/03/15/15:00', '2023/03/31/15:00', '2023/04/15/15:00', '2023/04/30/15:00',
    '2023/05/15/15:00', '2023/05/31/15:00', '2023/06/15/15:00', '2023/06/30/15:00',
    '2023/07/15/15:00', '2023/07/31/15:00', '2023/08/15/15:00', '2023/08/31/15:00',
    '2023/09/15/15:00', '2023/09/30/15:00', '2023/10/15/15:00', '2023/10/31/15:00',
    '2023/11/15/15:00', '2023/11/30/15:00', '2023/12/15/15:00', '2023/12/31/15:00',

    '2024/01/15/15:00', '2024/01/31/15:00', '2024/02/15/15:00', '2024/02/29/15:00',
    '2024/03/15/15:00', '2024/03/31/15:00', '2024/04/15/15:00', '2024/04/30/15:00',
    '2024/05/15/15:00', '2024/05/31/15:00', '2024/06/15/15:00', '2024/06/30/15:00',
    '2024/07/15/15:00', '2024/07/31/15:00', '2024/08/15/15:00', '2024/08/31/15:00',
    '2024/09/15/15:00', '2024/09/30/15:00', '2024/10/15/15:00', '2024/10/31/15:00',
    '2024/11/15/15:00', '2024/11/30/15:00', '2024/12/15/15:00', '2024/12/31/15:00',

    '2025/01/15/15:00', '2025/01/31/15:00', '2025/02/15/15:00', '2025/02/28/15:00',
    '2025/03/15/15:00', '2025/03/31/15:00', '2025/04/15/15:00', '2025/04/30/15:00',
    '2025/05/15/15:00', '2025/05/31/15:00', '2025/06/15/15:00', '2025/06/30/15:00',
    '2025/07/15/15:00', '2025/07/31/15:00', '2025/08/15/15:00', '2025/08/31/15:00',
    '2025/09/15/15:00', '2025/09/30/15:00', '2025/10/15/15:00', '2025/10/31/15:00',
    '2025/11/15/15:00', '2025/11/30/15:00', '2025/12/15/15:00', '2025/12/31/15:00',
]

term = ''

folder = "앙상블실험/"

result_path = "eval_reflection/eval_reflection_" + term + "_losscut" + str(conf.loss_cut) + ".csv"

hit_ratios = [0.2, 0.3, 0.4, 0.5]
eval_terms = [10, 20, 30, 40]
eval_widths = [10, 20, 30, 40, 50, 60, 70]

every_term_random = False
bayesian = False

margin = 10000000

def set_path(folder_name, result_folder):
    folder = folder_name + "/" + folder_name + "_" + term + "_losscut" + str(conf.loss_cut)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    result_path = result_folder + "/" + result_folder + "_" + term + "_losscut" + str(conf.loss_cut) + ".csv"

    return folder, result_path

def set_term(conf, last_trains, start_times, end_times, add_txt):
    start_index = start_times.index(conf.start_time)
    end_index = end_times.index(conf.end_time)

    last_trains = last_trains[start_index:end_index+1]
    start_times = start_times[start_index:end_index+1]
    end_times = end_times[start_index:end_index+1]

    term = datetime.strptime(conf.start_time, "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d") + "~" + \
           datetime.strptime(conf.end_time, "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d") + add_txt

    return last_trains, start_times, end_times, term

# 빈 데이터프레임 생성
model_pools = ["5C", "5HL", "5P", "10C", "10HL", "10P", "15C", "15HL", "15P", "20C", "20HL", "20P",
          "25C", "25HL", "25P", "30C", "30HL", "30P", "40C", "40HL", "40P"]

models = [

]

def main(conf, ep):
    ensemble_models = ""
    for i in range(len(conf.selected_model_types)):
        ensemble_models = ensemble_models + conf.selected_model_types[i] + "_"

    path = ep.folder + "/9시거래없음_ensemble_" + str(conf.pred_term) + conf.target_type + "_reinfo_" + \
           str(conf.reinfo_th) + "_width_" + str(conf.reinfo_width) + "_" + ensemble_models + ".xlsx"

    # 모델별 각 구간별 수익률 계산, 저장
    profit_sum = []
    #profit_product = 1
    profits = np.empty(0)
    dates = np.empty(0)
    closes = np.empty(0)
    for i in range(len(ep.last_trains)):
        conf.last_train = ep.last_trains[i]
        conf.start_time = ep.start_times[i]
        conf.end_time = ep.end_times[i]

        data.set_path(conf)

        if ep.every_term_random:

            conf.selected_model_types = sorted(random.sample(conf.model_pools, conf.selected_num))
            conf.reinfo_th = random.randint(0, 32) / 100#random.sample(hit_ratios, 1)[0]
            conf.pred_term = random.randrange(0, 41, 5)#random.sample(eval_terms, 1)[0]
            conf.reinfo_width = random.randrange(0, 71, 10)#random.sample(eval_widths, 1)[0]


        data.preprocessing(conf)

        conf.selected_checkpoint_path = ['' for i in range(conf.selected_num)]
        for j in range(conf.selected_num):
            conf.selected_checkpoint_path[j] = conf.last_train + "/60M_" + conf.selected_model_types[j] + "_best"

        p = ep.predict(conf)
        print(last_trains[i] + " 수익률: " + str(p))
        profit_sum.append(p)
        #profit_product *= p

        df = pd.read_csv(conf.result_path, encoding='euc-kr')
        profit = df['profit'].values - df['fee'].values
        profits = np.concatenate([profits, profit], axis=0)
        dates = np.concatenate([dates, df['date'].values], axis=0)
        closes = np.concatenate([closes, df['close'].values], axis=0)

        if ep.bayesian:

            # bayesian optimization
            bs.conf.last_train = last_trains[i]
            bs.conf.start_time = start_times[i]
            bs.conf.end_time = end_times[i]
            data.set_path(bs.conf)
            bs.conf.reinfo_th = conf.reinfo_th
            data.set_ensemble(bs.conf,  conf.selected_model_types)
            bs.conf.gubun = 0
            best = bs.best_params()

            # set up with best params
            conf.reinfo_th = float(best['params']['reinfo'])
            conf.pred_term = int(best['params']['term'])
            conf.loss_cut = float(best['params']['loss_cut'])
            conf.reinfo_width = int(best['params']['width'])

    profit_rates = profits.cumsum() / margin + 1
    close_rates = np.array(closes) / closes[0]
    profit_product = np.array(profit_sum).prod()
    #profit_product = np.array(profit_sum).mean()/np.array(profit_sum).std()

    print("최종 수익률", profit_rates[-1])
    print("복리누적: " + str(profit_product))
    MDD = calc_MDD(profit_rates)
    print("NDD: ", MDD)

    #dic = {'dates': dates, 'profits': profits, 'closes': closes, 'profit_rates': profit_rates}
    #pd.DataFrame(dic).to_excel(path, index=False, encoding='euc-kr')

    print(path)

    return dates, closes, profits, close_rates, profit_rates, MDD, profit_product, ensemble_models, path

# 주어긴 기간에서 walk forward 방식의 수익률 평가가 아닌 n개의 random 구간에서 평가
def main_random(conf, ep):
    ensemble_models = ""
    for i in range(len(conf.selected_model_types)):
        ensemble_models = ensemble_models + conf.selected_model_types[i] + "_"

    path = ep.folder + "/9시거래없음_ensemble_" + str(conf.pred_term) + conf.target_type + "_reinfo_" + \
           str(conf.reinfo_th) + "_width_" + str(conf.reinfo_width) + "_" + ensemble_models + ".xlsx"

    # 모델별 각 구간별 수익률 계산, 저장
    profit_sum = []
    #profit_product = 1
    profits = np.empty(0)
    dates = np.empty(0)
    closes = np.empty(0)
    avg_rates = []
    for k in range(len(ep.last_trains)-1):
        conf.last_train = ep.last_trains[k]
        for n in range(3):
            if n == 0:
                i = k
            else:
                i = random.randint(k, len(ep.last_trains)-1)
            conf.start_time = ep.start_times[i]
            conf.end_time = ep.end_times[i]

            data.set_path(conf)

            data.preprocessing(conf)

            conf.selected_checkpoint_path = ['' for i in range(conf.selected_num)]
            for j in range(conf.selected_num):
                conf.selected_checkpoint_path[j] = conf.last_train + "/60M_" + conf.selected_model_types[j] + "_best"

            p = ep.predict(conf)
            profit_sum.append(p)
            print("-----" + last_trains[i] + " 수익률: " + str(p))

            if n == 0:
                print(last_trains[k] + " 수익률: " + str(p))

                df = pd.read_csv(conf.result_path, encoding='euc-kr')
                profit = df['profit'].values - df['fee'].values
                profits = np.concatenate([profits, profit], axis=0)

                dates = np.concatenate([dates, df['date'].values], axis=0)
                closes = np.concatenate([closes, df['close'].values], axis=0)

            elif n == 2:
                avg_rates.append(np.array(profit_sum[-3:]).mean())

    profit_rates = profits.cumsum() / margin + 1
    close_rates = np.array(closes) / closes[0]
    profit_product = np.array(profits / margin + 1).prod()

    print("최종 수익률" + str(profit_rates[-1]) + " vs. " + str((np.array(avg_rates) - 1).sum()))
    print("복리누적: " + str(profit_product) + " vs. " + str(np.array(avg_rates).prod()))
    MDD = calc_MDD(np.array(avg_rates))
    print("MDD: ", MDD)

    print(path)

    return dates, close_rates, profit_rates, close_rates, profit_rates, MDD, profit_product, ensemble_models, path

def predict(conf):

    #개별 모델 예측 전 앙상블 pred_term, target_type save하고 개별 모델 예측후 복원
    save_term = conf.pred_term
    save_type = conf.target_type

    #모델들의 예측값 생성
    model = md.create_model(conf)
    r = []
    for i in range(conf.selected_num):

        model.load_weights(conf.selected_checkpoint_path[i])

        model_pred_term, mmodel_target_type = parsing(conf.selected_checkpoint_path[i])
        conf.pred_term = model_pred_term
        conf.target_type = mmodel_target_type

        pred = md.predict(model, conf)[:, 1]

        r.append(pred)

    r = np.array(r)

    conf.pred_term = save_term
    conf.target_type = save_type

    #앙상블의 예측값 생성
    pred = []
    for i in range(len(r[0])):
        cnt = [0, 0, 0]
        for j in range(conf.selected_num):
            cnt[r[j][i]] += 1
        if cnt[1] == cnt[2] and cnt[1] >= cnt[0]:
            pred.append(0)
        else:
            pred.append(np.argmax(cnt))

    # 시가, 고가, 저가, 종가 검색
    df = pd.read_csv(conf.df0_path, encoding='euc-kr')

    if conf.start_time > df['date'].values[-1]:
        conf.start_time = df['date'].values[-1]

    start_index = df.loc[df['date'] >= conf.start_time].index.min()
    end_index = df.loc[df['date'] <= conf.end_time].index.max()
    dates = df['date'].values[start_index:end_index + 1]
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]
    #  0: 고점, 1: 저점

    pred_results = []
    for i in range(len(pred)):
        pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])
    pred_results = np.array(pred_results)

    if conf.reinfo_th > 0:
        mr.th = conf.reinfo_th
        mr.pred_term = conf.pred_term
        mr.target_type = conf.target_type
        pred = mr.reinfo(pred, pred_results, conf.start_time, conf.reinfo_width)
        pred_results[:, 1] = np.array(pred)


    # 결과 파일에 저장
    # 0: 정상, 1: 고점 2:저점
    pd.DataFrame(pred_results, columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(conf.result_path, index=False, encoding='euc-kr')

    # 수익률 계산하여 return
    profit.loss_cut = conf.loss_cut
    profit.result_path = conf.result_path
    return profit.calc_profit()

def parsing(path):

    start = path.find('/60M_') + 5
    end = path.find('_best')
    model_type = path[start:end]

    c = model_type.find('C')
    h = model_type.find('HL')
    p = model_type.find('P')
    if c != -1:
        pred_term = int(model_type[:c])
        target_type = 'C'
    elif h != -1:
        pred_term = int(model_type[:h])
        target_type = 'HL'
    elif p != -1:
        pred_term = int(model_type[:p])
        target_type = 'P'
    else:
        print("argument error " + model_type)
        exit(0)

    return pred_term, target_type

def calc_MDD(profit_rates):

    MDD = []
    for i in range(len(profit_rates)):
        MDD.append((profit_rates[:i+1].max() - profit_rates[i]) / profit_rates[:i+1].max())
    return np.array(MDD).max()