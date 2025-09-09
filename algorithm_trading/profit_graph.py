import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sys

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def run(start_date, current_price):
    df = pd.read_csv("trans.csv", encoding='euc-kr')
    df = df.loc[df['거래일시']>=start_date]
    dates = df['거래일시'].values
    start_date = df['거래일시'].min()
    gubun = df['매매구분'].values
    purchase_price = df['거래가격'].values
    liquid_price = df['청산가격'].values
    if int(liquid_price[-1]) == 0:
        liquid_price[-1] = purchase_price[-1]

    seq = []
    profits = []
    accu_rates = []
    accu_buy_and_hold = []
    accu_sell_and_hold = []
    for i in range(len(dates)):
        if liquid_price[i] == 0:
            liquid_price[i] = current_price

        #if i % 20 == 0:
        #    seq.append(str(dates[i][:10]))
        #else:
        #    seq.append('')

        seq.append(i)

        # 앙상블 전략 손익
        if gubun[i] == '매수':
            profits.append((liquid_price[i] - purchase_price[i])*250000 - purchase_price[i]*250000*0.00006)
        else:
            profits.append((purchase_price[i] - liquid_price[i]) * 250000 - purchase_price[i]*250000*0.00006)
        accu_rates.append(sum(profits[:i+1])/(purchase_price.mean()*250000*0.078) + 1)

        # buy and hold 전략 손익
        if i == 0:
            accu_buy_and_hold.append((liquid_price[0] - purchase_price[0]) * 13.3 / purchase_price[0] + 1)
        else:
            accu_buy_and_hold.append((liquid_price[i]-liquid_price[0])*13.3/liquid_price[0]+1)

        # sell and hol 전략 손익
        if i == 0:
            accu_sell_and_hold.append((purchase_price[0] - liquid_price[0]) * 13.3 / purchase_price[0] + 1)
        else:
            accu_sell_and_hold.append((liquid_price[0]-liquid_price[i])*13.3/liquid_price[0]+1)

    #dates = dates[:len(seq)]

    plt.xlabel('날짜')
    plt.ylabel('수익률')
    plt.title(start_date[:10] + '~' + dates[-1] + ' 수익률: ' + str(round((accu_rates[-1]-1)*100, 3)) + '%')
    plt.plot(dates, accu_rates, label='3개의 딥러닝 모델들의 앙상블 전략')#, 'ro')
    plt.plot(dates, accu_buy_and_hold, label='선물지수등락 with lev. 13.3')#, 'bo')
    plt.legend(loc='upper left')

    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.xticks(rotation=30)

    #plt.plot(seq, accu_sell_and_hold)#, 'go')
    #plt.hist(accu_rates, bins=100, density=True, alpha=0.7, histtype='stepfilled')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        run(sys.argv[1], 0)
    else:
        run('2022/01/01/00:00', 0)
    sys.exit(0)