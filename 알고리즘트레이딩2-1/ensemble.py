# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 거래에 실재 적용되는 앙상블 모델의 prediction 결과와 수익률

import data
from data import config
conf = config()
import ensemble_proc as ep
import profit

import pandas as pd
import datetime

import make_reinfo3 as mr
ep.mr = mr

import tensorflow as tf
from tensorflow import keras
import numpy as np

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
            self,
            num_outputs,
            num_hidden_units):
        """Initialize."""
        super().__init__()

        self.common1 = keras.layers.Dense(num_hidden_units, activation="relu")
        self.common2 = keras.layers.Dense(int(num_hidden_units / 2), activation="relu")
        self.actor = keras.layers.Dense(num_outputs, activation="softmax")
        self.critic = keras.layers.Dense(1)

    def call(self, inputs):
        x1 = self.common1(inputs)
        x2 = self.common2(x1)
        return self.actor(x2), self.critic(x2)

# A2C 모델 초기화
#actor_critic = ActorCritic(5, 128)
#actor_critic = tf.keras.models.load_model("H:/actor_critic_reinfo/알고리즘2-1/2022-01-01~2025-01-31")

# 6월 27일 the best of random mr3 1000 (3_6_12개월), 1.0225344539539902
#model = ['15C', '25C', '25HL', 0.05, 20, 35, 0.0025]
# 7월 4일 the best of random mr3 1000 (3_6_12개월), 0.9533358758468654
#model = ['20C', '20HL', '25P', 0.05, 20, 70, 0.0015]
# 6월 27일 the best of random mr3 1000 (random3개월), 1.1327607666085375
#model = ["25P", "30HL", "5C", 0.05, 20, 25, 0.0035]
# 7월 18일 the best of random mr3 1000 (random3개월), 1.0107652315985216
#model = ['15C', '20HL', '25P', 0.05, 20, 15, 0.0025]
# 7월 25일 the best of random mr3 1000 (random3개월)
#model = ["10C", "30P", "40P", 0.25, 20, 65, 0.005]#random3_6_12개월['10C', '20C', '25P', 0.65, 20, 20, 0.0015]
# 8월 1일 the best of random mr3 1000 (random3개월_6개월_12개월), 1.0107652315985216
#model = ['10C', '15P', '30C', 0.3, 20, 15, 0.0025]
# 8월 15일 the best of random mr3 1000 (3_6_12개월), 0.9533358758468654
#model = ['15C', '20HL', '40P', 0.7, 20, 65, 0.0025]

# 8월 31일 the best of random mr3 1000 (3_6_12개월)
model = ['15C', '15HL', '20HL', 0.25, 20, 10, 0.003]

m = len(model)

data.set_start_end_time(conf, '2025/09/01/09:00', '2025/01/10/15:00', '2025-08-31')
data.set_path(conf)
data.set_target_type(conf, 'C')
data.set_pred_term(conf, model[m-3])
data.set_reinfo(conf, model[m-4]) # reinfo_th,
data.set_ensemble(conf, model[:m-4]) # selected_model_types
data.set_profit(conf, model[m-1], 1) # loss_cut, profit_cut
data.set_width(conf, model[m-2]) # eval_width

ep.profit.slippage = 0.05

if __name__ == "__main__":

    #ep.profit.trading_9h = True

    conf.end_time = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M")
    #conf.start_time = (datetime.datetime.strptime(conf.last_train, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime("%Y/%m/%d/09:00")
    #conf.start_time = (datetime.datetime.strptime(conf.end_time, '%Y/%m/%d/%H:%M') - datetime.timedelta(days=2)).strftime("%Y/%m/%d/09:00")

    #prediction only
    conf.gubun = 0
    conf.result_path = 'pred_83_results.csv'

    print('data processing start...')
    now = datetime.datetime.now()
    print(now)
    data.preprocessing(conf)
    print('data processing end...')
    r = ep.predict(conf)
    print("수익률: ", r)

    ############################################
    # actor critic으로 예측값 보정 후 수익률 출력
    ############################################
    #p = ep.a2c_reinfo(conf, ep.actor_critic)
    #print("actor-critic 보정후 수익률: ", str(p))

    print(pd.read_csv(conf.result_path).values[-7:, [0, 1, 5]])
