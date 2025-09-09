# Copyright 2022 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# all_term_ensemble을 9시시가에거래/9시거래없음, 앙상블 구성요소, 앙상블/모델 reinfo, pred_term, target_type
# 여러가지로 바꾸어가면서 실행행

import all_term_ensemble as ens
import pandas as pd
import numpy as np
import sys

folder = "앙상블실험4/"

trading_9h = [True]#, False]
#
selected_model_types = [['10P', '20C', '40HL'],
                        ['5C', '10P', '20HL', '30C', '40P'],
                        ["10HL", "25C", "25HL", "30C", "40P"]
                       ]
reinfo_th = [0.3, 0.4]
model_reinfo_th = [0.4, 0.3]
pred_term = [20, 30, 40]
target_type = ['C', 'P']

results = []
for tr in trading_9h:
    for sel in selected_model_types:
        for reinfo in reinfo_th:
            for m_reinfo in model_reinfo_th:
                for pt in pred_term:
                    for type in target_type:
                        ens.folder = folder
                        ens.trading_9h = tr
                        ens.selected_model_types = sel
                        ens.et.reinfo_th = reinfo
                        ens.et.model_reinfo_th = m_reinfo
                        ens.et.pred_term = pt
                        ens.et.target_type = type
                        _, _, _, r, _ = ens.main()
                        results.append(r)
pd.DataFrame(np.array(results), columns=['path', '평균', '복리']).to_excel("all_ensemble_results.xlsx", index=False, encoding='euc-kr')