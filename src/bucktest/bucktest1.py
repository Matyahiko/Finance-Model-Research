import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import talib

# S&P500 ETFのデータを取得
df = yf.download('SPY')

# 調整済み終値を抽出
spy = df['Adj Close']

# 移動平均を計算
ma1 = spy.rolling(5).mean()
ma2 = spy.rolling(60).mean()

# 移動平均をプロット
ma1.plot()
ma2.plot()

# バックテストの初期化
pnls = []
entry = None
dates = []
unrlzd = []
unrlzd_dates = []
inpos = 0
hold_times = []
COMMS = 0.005

# バックテストのメインループ
for i in range(len(spy)):
    unr = ((spy.iloc[i]-spy.iloc[i-1])/spy.iloc[i-1])*inpos
    unrlzd.append(unr)
    unrlzd_dates.append(spy.index[i])

    if (ma1.iloc[i]-ma2.iloc[i])>0 and (ma1.iloc[i-1]-ma2.iloc[i-1])<0:
        entry = spy.iloc[i]
        inpos = 1
        start = spy.index[i]
        unrlzd[-1] -= COMMS/2

    elif (ma1.iloc[i]-ma2.iloc[i])<0 and (ma1.iloc[i-1]-ma2.iloc[i-1])>0:
        if entry:
            pnl = (spy.iloc[i] - entry)/entry - COMMS
            pnls.append(pnl)
            dates.append(spy.index[i])
            hold_times.append((spy.index[i]-start).total_seconds())
            unrlzd[-1] -= COMMS/2

        entry = spy.iloc[i]
        inpos = 0
        start = spy.index[i]

# 戦略のリターンをプロット
plt.plot(dates, np.cumprod(1+np.array(pnls)),'-o')
plt.plot(unrlzd_dates,np.cumprod(1+np.array(unrlzd)))

# リスク調整後リターン（シャープレシオ）を計算
rar = np.mean(unrlzd)/np.std(unrlzd)*np.sqrt(252)
print(f"Risk-adjusted return: {rar:.4f}")

# トレードあたりの利益を計算
ppt = np.mean(pnls)
print(f"Profit per trade: {ppt:.4f}")

# 平均保有期間を計算
hold_days = np.array(hold_times)/86400
print(f"Average holding period (days): {np.mean(hold_days):.2f}")

# 戦略のリターンとPnLのデータフレームを作成
df_rets = pd.DataFrame(unrlzd, index=unrlzd_dates)
df_pnl = (1+df_rets).cumprod()

# ドローダウンを計算してプロット
(df_pnl/df_pnl.expanding().max()-1).plot()

# ベータを計算
beta = np.corrcoef(spy.pct_change().iloc[2:],df_rets[0].iloc[2:])[0,1]
print(f"Beta: {beta:.4f}")

# TA-Libのインストールとインポート
# 注: Google Colabでは事前にTA-Libをインストールする必要があります

# TA-Libを使用して移動平均を計算しプロット
ma50 = talib.SMA(spy,50)
ma200 = talib.SMA(spy,200)
plt.plot(ma50)
plt.plot(ma200)

# TA-Libのすべての関数を表示
print(dir(talib))

# TA-Libを使用してテクニカル指標を計算
crows = talib.CDL3BLACKCROWS(df.Open,df.High,df.Low,df.Close)
crows.plot()
