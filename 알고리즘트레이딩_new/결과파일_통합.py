# -*- coding:utf-8 -*-

import os
import pandas as pd
import openpyxl

path = '앙상블실험3/앙상블실험3_2023-01-01~2023-06-16_random_losscut0.01'
file_list = os.listdir(path)
os.chdir(path)
print(file_list[0])
df = pd.read_excel(file_list[0])[['dates', 'profits']]
for file_path in file_list:
    data = pd.read_excel(file_path)
    df = pd.concat([df, data['profits']], axis=1)

df = df.reset_index(drop=True).to_excel("total.xlsx", index=False)