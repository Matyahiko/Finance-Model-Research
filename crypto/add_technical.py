import json
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split

# JSONファイルを読み込む
with open("raw_data/btc/BTC-JPY_5min_2021-2024_split.json", "r") as f:
    data = json.load(f)

# データフレームを作成
df = pd.DataFrame(data)

# # 日時をパースして設定
# df["close_time"] = pd.to_datetime(df["close_time"], format="%Y/%m/%d %H:%M:%S")
# df.set_index("close_time", inplace=True)

# 日時をパースして設定
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
df.set_index("close_time", inplace=True)

# テクニカル指標を計算
open_price, high_price, low_price, close_price = df["open_price"], df["high_price"], df["low_price"], df["close_price"]

# 移動平均
df["SMA5"] = talib.SMA(close_price, timeperiod=5)
df["SMA10"] = talib.SMA(close_price, timeperiod=10)
df["SMA20"] = talib.SMA(close_price, timeperiod=20)
df["SMA50"] = talib.SMA(close_price, timeperiod=50)
df["SMA100"] = talib.SMA(close_price, timeperiod=100)
df["SMA200"] = talib.SMA(close_price, timeperiod=200)

# ボリンジャーバンド
df["upper_band"], df["middle_band"], df["lower_band"] = talib.BBANDS(close_price, timeperiod=20)

# MACD
df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)

# MACDクロスを計算
df["macd_cross"] = np.where((df["macd"].shift(1) < df["macdsignal"].shift(1)) & (df["macd"] > df["macdsignal"]), 1, 0)
df["macd_cross"] = np.where((df["macd"].shift(1) > df["macdsignal"].shift(1)) & (df["macd"] < df["macdsignal"]), -1, df["macd_cross"])

# RSI
df["RSI"] = talib.RSI(close_price, timeperiod=14)

# ストキャスティクス
df["slowk"], df["slowd"] = talib.STOCH(high_price, low_price, close_price, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# ADX
df["ADX"] = talib.ADX(high_price, low_price, close_price, timeperiod=14)

# CCI
df["CCI"] = talib.CCI(high_price, low_price, close_price, timeperiod=14)

# ATR
df["ATR"] = talib.ATR(high_price, low_price, close_price, timeperiod=14)

# ROC
df["ROC"] = talib.ROC(close_price, timeperiod=10)

# Williams %R
df["Williams %R"] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)

# 値上がり率（単純収益率）を計算
df["return"] = df["close_price"].pct_change()

# 欠損値を削除
df.dropna(inplace=True)

# 学習データとテストデータに分割
train_size = 0.8  # 学習データの割合を指定
train_df = df[:int(len(df) * train_size)]
test_df = df[int(len(df) * train_size):]

print(f"全データ数: {len(df)}")
print(f"学習データ数: {len(train_df)}")
print(f"テストデータ数: {len(test_df)}")

# CSVファイルに出力
train_df.to_csv("/root/src/src/crypto/procesed/btc_5min_technical_analysis_train.csv")
test_df.to_csv("/root/src/src/crypto/procesed/btc_5min_technical_analysis_test.csv")

