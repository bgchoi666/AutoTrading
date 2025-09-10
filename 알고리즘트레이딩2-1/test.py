import pandas as pd
import numpy as np
import random

a = pd.DataFrame(columns=['a', 'b'], dtype=(int, int))

pred_term = 5

df = pd.read_csv("kospi200f_11_60M.csv", encoding='euc-kr')

rates = np.array(df["종가"].rolling(window=pred_term + 1).apply(lambda x: x[pred_term] - x[0]))

for i in range(pred_term):
    rates[i] = 0

a = rates[np.where(rates >0)].mean()

exit(0)