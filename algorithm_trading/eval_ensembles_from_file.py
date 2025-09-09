import pandas as pd
import ensemble_test as ep
import sys
import random

selected_num = 3

file_name = "test/ensemble_results_08-02~08-20.csv"

def run():
    ens = pd.read_csv(file_name, encoding='euc-kr')

    rate = []
    cnt = 0
    for i in range(len(ens)):
        e = ens.values[i, :ep.selected_num]
        print(e)

        for j in range(ep.selected_num):
            ep.selected_checkpoint_path[j+1] = ep.checkpoint_path[int(e[j])]

        rate.append(str(ep.predict()))

        cnt += 1
        if cnt % 100 == 0:
            print("========================================================================")
            print(" ")
            print(cnt)
            print(" ")
            print("========================================================================")
    ens['7월수익률'] = rate

    ens.to_csv(file_name, index=False, encoding='euc-kr')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('error!!!  not enough arguments.')
        sys.exit(1)
    selected_num = int(sys.argv[1])
    ep.selected_num = selected_num
    ep.selected_checkpoint_path = ['' for i in range(ep.selected_num + 1)]
    ep.model = ep.create_model(3)
    run()

    sys.exit(0)