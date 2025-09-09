import pandas as pd
import numpy as np
import ensemble_test as ep
import sys
import random

selected_num = 3
pred_term = 5
norm_term = 20

file_name = "test/ensemble-of-ensembles/ensemble_results_07.csv"
df0_path = df0_path = 'kospi200f_11_60M.csv'
df_pred_path = 'test/kospi200f_60M_pred.csv'
result_path = 'test/ensemble-of-ensembles/ensemble_results.csv'

start_time = '2021/07/01/09:00'
end_time = '2021/07/30/14:00'

start_time2 = '2021/08/02/09:00'
end_time2 = '2021/08/27/14:00'

input_size = 83

# 시가, 고가, 저가, 종가 검색
df = pd.read_csv(df0_path, encoding='euc-kr')
start_index = df.loc[df['date'] <= start_time].index.max()
end_index = df.loc[df['date'] <= end_time].index.max()
high = df['고가'].values[start_index:end_index + 1]
low = df['저가'].values[start_index:end_index + 1]
close = df['종가'].values[start_index:end_index + 1]
open = df['시가'].values[start_index:end_index + 1]

# 시가, 고가, 저가, 종가 검색
start_index2 = df.loc[df['date'] <= start_time2].index.max()
end_index2 = df.loc[df['date'] <= end_time2].index.max()
high2 = df['고가'].values[start_index2:end_index2 + 1]
low2 = df['저가'].values[start_index2:end_index2 + 1]
close2 = df['종가'].values[start_index2:end_index2 + 1]
open2 = df['시가'].values[start_index2:end_index2 + 1]

def preprocessing():
    norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')

    start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
    last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

    if last_date >= end_time2 and start_date <= start_time:
        print('nothing done! in this preprocessing')
        return

    df0 = pd.read_csv(df0_path, encoding='euc-kr')

    df0["시가대비종가변화율"] = (df0["종가"] - df0["시가"])/df0["시가"]*100
    df0["시가대비고가변화율"] = (df0["고가"] - df0["시가"])/df0["시가"]*100
    df0["시가대비저가변화율"] = (df0["저가"] - df0["시가"])/df0["시가"]*100
    df0["종가대비고가변화율"] = (df0["고가"] - df0["종가"])/df0["종가"]*100
    df0["종가대비저가변화율"] = (df0["저가"] - df0["종가"])/df0["종가"]*100

    df0["1일전"] = np.concatenate([[0], df0["종가"].values[:-1]])
    df0["2일전"] = np.concatenate([[0, 0], df0["종가"].values[:-2]])
    df0["3일전"] = np.concatenate([[0, 0, 0], df0["종가"].values[:-3]])
    df0["4일전"] = np.concatenate([[0, 0, 0, 0], df0["종가"].values[:-4]])
    df0["5일전"] = np.concatenate([[0, 0, 0, 0, 0], df0["종가"].values[:-5]])

    df0["6일전"] = np.concatenate([[0, 0, 0, 0, 0, 0], df0["종가"].values[:-6]])
    df0["7일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-7]])
    df0["8일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-8]])
    df0["9일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-9]])
    df0["10일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-10]])

    df0["1일수익률"] = df0["종가"].rolling(window=2).apply(lambda x: x[1] - x[0])
    df0["3일수익률"] = df0["종가"].rolling(window=4).apply(lambda x: x[3] - x[0])
    df0["5일수익률"] = df0["종가"].rolling(window=6).apply(lambda x: x[5] - x[0])
    df0["10일수익률"] = df0["종가"].rolling(window=11).apply(lambda x: x[10] - x[0])
    df0["20일수익률"] = df0["종가"].rolling(window=21).apply(lambda x: x[20] - x[0])
    df0["40일수익률"] = df0["종가"].rolling(window=41).apply(lambda x: x[40] - x[0])
    df0["60일수익률"] = df0["종가"].rolling(window=61).apply(lambda x: x[60] - x[0])
    df0["90일수익률"] = df0["종가"].rolling(window=91).apply(lambda x: x[90] - x[0])
    df0["120일수익률"] = df0["종가"].rolling(window=121).apply(lambda x: x[120] - x[0])
    df0["180일수익률"] = df0["종가"].rolling(window=181).apply(lambda x: x[180] - x[0])
    df0["240일수익률"] = df0["종가"].rolling(window=241).apply(lambda x: x[240] - x[0])


    df0["5일최고"] = df0["고가"].rolling(window=5).max()
    df0["20일최고"] = df0["고가"].rolling(window=20).max()
    df0["60일최고"] = df0["고가"].rolling(window=60).max()
    df0["120일최고"] = df0["고가"].rolling(window=120).max()
    df0["240일최고"] = df0["고가"].rolling(window=240).max()

    df0["5일최저"] = df0["저가"].rolling(window=5).min()
    df0["20일최저"] = df0["저가"].rolling(window=20).min()
    df0["60일최저"] = df0["저가"].rolling(window=60).min()
    df0["120일최저"] = df0["저가"].rolling(window=120).min()
    df0["240일최저"] = df0["저가"].rolling(window=240).min()

    df0["1일전거래량"] = np.concatenate([[0], df0["거래량"].values[:-1]])
    df0["2일전거래량"] = np.concatenate([[0, 0], df0["거래량"].values[:-2]])
    df0["3일전거래량"] = np.concatenate([[0, 0, 0], df0["거래량"].values[:-3]])
    df0["4일전거래량"] = np.concatenate([[0, 0, 0, 0], df0["거래량"].values[:-4]])
    df0["5일전거래량"] = np.concatenate([[0, 0, 0, 0, 0], df0["거래량"].values[:-5]])

    df0["6일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0], df0["거래량"].values[:-6]])
    df0["7일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-7]])
    df0["8일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-8]])
    df0["9일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-9]])
    df0["10일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-10]])

    df0["1일거래변화량"] = df0["거래량"].rolling(window=2).apply(lambda x: x[1] - x[0])
    df0["3일거래변화량"] = df0["거래량"].rolling(window=4).apply(lambda x: x[3] - x[0])
    df0["5일거래변화량"] = df0["거래량"].rolling(window=6).apply(lambda x: x[5] - x[0])
    df0["10일거래변화량"] = df0["거래량"].rolling(window=11).apply(lambda x: x[10] - x[0])
    df0["20일거래변화량"] = df0["거래량"].rolling(window=21).apply(lambda x: x[20] - x[0])
    df0["40일거래변화량"] = df0["거래량"].rolling(window=41).apply(lambda x: x[40] - x[0])
    df0["60일거래변화량"] = df0["거래량"].rolling(window=61).apply(lambda x: x[60] - x[0])
    df0["90일거래변화량"] = df0["거래량"].rolling(window=91).apply(lambda x: x[90] - x[0])
    df0["120일거래변화량"] = df0["거래량"].rolling(window=121).apply(lambda x: x[120] - x[0])
    df0["180일거래변화량"] = df0["거래량"].rolling(window=181).apply(lambda x: x[180] - x[0])
    df0["240일거래변화량"] = df0["거래량"].rolling(window=241).apply(lambda x: x[240] - x[0])

    df0["5일평균거래량"] = df0["거래량"].rolling(window=5).mean()
    df0["20일평균거래량"] = df0["거래량"].rolling(window=20).mean()
    df0["60일평균거래량"] = df0["거래량"].rolling(window=60).mean()
    df0["120일평균거래량"] = df0["거래량"].rolling(window=120).mean()
    df0["240일평균거래량"] = df0["거래량"].rolling(window=240).mean()

    df0["5일최고거래량"] = df0["거래량"].rolling(window=5).max()
    df0["20일최고거래량"] = df0["거래량"].rolling(window=20).max()
    df0["60일최고거래량"] = df0["거래량"].rolling(window=60).max()
    df0["120일최고거래량"] = df0["거래량"].rolling(window=120).max()
    df0["240일최고거래량"] = df0["거래량"].rolling(window=240).max()

    df0["5일최저거래량"] = df0["거래량"].rolling(window=5).min()
    df0["20일최저거래량"] = df0["거래량"].rolling(window=20).min()
    df0["60일최저거래량"] = df0["거래량"].rolling(window=60).min()
    df0["120일최저거래량"] = df0["거래량"].rolling(window=120).min()
    df0["240일최저거래량"] = df0["거래량"].rolling(window=240).min()

    #df0.to_csv(df_raw_path, encoding='euc-kr')

    start_index = df0.loc[df0['date'] <= start_time].index.max()
    end_index = df0.loc[df0['date'] <= end_time2].index.max()

    df = df0[start_index - (norm_term-1) : end_index + 1].reset_index(drop=True)
    norm_df = df[norm_term-1 : ].reset_index(drop=True).copy()

    for i in range(end_index - start_index + 1):
        for j in range(1, input_size+1):
            m = df.iloc[i:i+norm_term, j].mean()
            s = df.iloc[i:i+norm_term, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i+norm_term-1, j] - m) / s
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')

def run():
    preprocessing()

    # create prediction input values
    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')
    start_index = df_pred.loc[df_pred['date'] <= start_time].index.max()
    end_index = df_pred.loc[df_pred['date'] <= end_time].index.max()
    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)
    pred_input = df_pred.values[start_index:end_index + 1, :input_size].reshape(-1, input_size)

    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')
    start_index2 = df_pred.loc[df_pred['date'] <= start_time2].index.max()
    end_index2 = df_pred.loc[df_pred['date'] <= end_time2].index.max()
    dates2 = df_pred.pop('date').values[start_index2:end_index2 + 1].reshape(-1)
    pred_input2 = df_pred.values[start_index2:end_index2 + 1, :input_size].reshape(-1, input_size)

    ens = pd.read_csv(file_name, encoding='euc-kr')

    rate = []
    rate2 = []
    total_models = []
    cnt = 0
    for n in range(1000):
        ensemble_no = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], selected_num)

        model_no = []
        for j in range(selected_num):
            for k in range(selected_num):
                model_no.append(int(ens.values[ensemble_no[j], k]))
        total_models.append(model_no)

        pred = []
        pred2 = []
        for n in range(selected_num):
            e = ens.values[ensemble_no[n], :selected_num]
            print(e)

            for j in range(ep.selected_num):
                ep.selected_checkpoint_path[j+1] = ep.checkpoint_path[int(e[j])]

            r = []
            for i in range(1, selected_num + 1):
                ep.model.load_weights(ep.selected_checkpoint_path[i])
                p = ep.model.predict(pred_input)
                r.append(np.argmax(p, axis=1).reshape(-1))
            r = np.array(r)

            r2 = []
            for i in range(1, selected_num + 1):
                ep.model.load_weights(ep.selected_checkpoint_path[i])
                p = ep.model.predict(pred_input2)
                r2.append(np.argmax(p, axis=1).reshape(-1))
            r2 = np.array(r2)

            pred_temp = []
            for i in range(len(r[0])):
                if list(r[:, i]).count(2) >= int(selected_num / 2 + 1):
                    pred_temp.append(2)
                elif list(r[:, i]).count(1) >= int(selected_num / 2 + 1):
                    pred_temp.append(1)
                elif list(r[:, i]).count(0) >= int(selected_num / 2 + 1):
                    pred_temp.append(0)
                else:
                    pred_temp.append(0)
            pred.append(pred_temp)

            pred2_temp = []
            for i in range(len(r2[0])):
                if list(r2[:, i]).count(2) >= int(selected_num / 2 + 1):
                    pred2_temp.append(2)
                elif list(r2[:, i]).count(1) >= int(selected_num / 2 + 1):
                    pred2_temp.append(1)
                elif list(r2[:, i]).count(0) >= int(selected_num / 2 + 1):
                    pred2_temp.append(0)
                else:
                    pred2_temp.append(0)
            pred2.append(pred2_temp)

        print(str(cnt) + " =======================================================")

        e_pred = []
        for i in range(len(pred[0])):
            if [pred[0][i], pred[1][i], pred[2][i]].count(2) >= int(selected_num / 2 + 1):
                e_pred.append(2)
            elif [pred[0][i], pred[1][i], pred[2][i]].count(1) >= int(selected_num / 2 + 1):
                e_pred.append(1)
            else:
                e_pred.append(0)
        rate.append(str(calc_profit(e_pred, open, high, low, close, dates)))

        e_pred = []
        for i in range(len(pred2[0])):
            if [pred2[0][i], pred2[1][i], pred2[2][i]].count(2) >= int(selected_num / 2 + 1):
                e_pred.append(2)
            elif [pred2[0][i], pred2[1][i], pred2[2][i]].count(1) >= int(selected_num / 2 + 1):
                e_pred.append(1)
            else:
                e_pred.append(0)
        rate2.append(str(calc_profit(e_pred, open2, high2, low2, close2, dates2)))

        cnt += 1
        if cnt % 100 == 0:
            print("========================================================================")
            print(" ")
            print(cnt)
            print(" ")
            print("========================================================================")
    total_models = np.array(total_models).T
    dic = {'model1': total_models[0], 'model2': total_models[1], 'model3': total_models[2], 'model4': total_models[3],
           'model5': total_models[4], 'model6': total_models[5], 'model7': total_models[6], 'model8': total_models[7],
           'model9': total_models[8], 'rate1': rate, 'rate2': rate2}
    pd.DataFrame(dic).to_csv(result_path, index=False, encoding='euc-kr')

def calc_profit(pred, open, high, low, close, dates):

    # 15시 종가를 익일 시가로 조정
    for i in range(len(dates)-1):
        if dates[i][11:13] == '15':
            close[i] = open[i+1]


    # 손익, 수수료 계산
    state = 0
    count = 0
    buy_price = 0
    profit = []
    fee = []

    for i in range(len(dates)):

        p = int(pred[i])
        c = float(close[i])
        h = float(high[i])
        l = float(low[i])

        if state == 1:
            if h - buy_price > buy_price * 0.015:
                profit.append(-buy_price*0.015)
                fee.append(buy_price*250000*0.00003)
                state = p
                count = 1
                buy_price = c
            elif count == pred_term-1 or p == 2:
                profit.append(buy_price - c)
                fee.append(buy_price*250000*0.00003)
                state = p
                count = 1
                buy_price = c
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                count %= pred_term
        elif state == 2:
            if buy_price - l > buy_price * 0.015:
                profit.append(-buy_price*0.015)
                fee.append(buy_price*250000*0.00003)
                state = p
                count = 1
                buy_price = c
            elif count == pred_term-1 or p == 1:
                profit.append(c - buy_price)
                fee.append(buy_price*250000*0.00003)
                state = p
                count = 1
                buy_price = c
            else:
                profit.append(0)
                fee.append(0)
                count += 1
                count %= pred_term
        else:
            if p == 1 or p == 2:
                state = p
                count = 1
                buy_price = c
            else:
                count = 0
            profit.append(0)
            fee.append(0)


    # 결과 파일에 저장, 수익률 계산하여여 reurn
    #print(" 0: 정상, 1: 고점 2:저점")
    #pred_results = []
    #for i in range(len(dates)):
    #    pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i], profit[i], fee[i]])

    #pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close', 'profit', 'fee']).to_csv(result_path, index=False, encoding='euc-kr')

    return (sum(profit)*250000 - sum(fee))/((close[0]+close[len(profit)-1])/2*250000*0.08)+1

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