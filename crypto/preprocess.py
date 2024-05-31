import json
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# JSONファイルを読み込む
with open('raw_data/btc/BTC-JPY_5min_2021-2024.json', 'r') as f:
    data = json.load(f)

# データフレームを作成
df = pd.DataFrame(data)

# 日時をパースして設定
df['close_time'] = pd.to_datetime(df['close_time'], format='%Y/%m/%d %H:%M:%S')

# 分割するデータポイント数を指定
split_point = 1000  # 例として100000を指定

# 指定したデータポイント数で分割
df_split = df[:split_point]

print(f"全データ数: {len(df)}")
print(f"分割後のデータ数: {len(df_split)}")

# 日時をパースして設定
df_split["close_time"] = pd.to_datetime(df_split["close_time"], unit="ms")
df_split.set_index("close_time", inplace=True)

# テクニカル指標を計算
open_price, high_price, low_price, close_price = df_split["open_price"], df_split["high_price"], df_split["low_price"], df_split["close_price"]

# 移動平均
df_split["SMA5"] = talib.SMA(close_price, timeperiod=5)
df_split["SMA10"] = talib.SMA(close_price, timeperiod=10)
df_split["SMA20"] = talib.SMA(close_price, timeperiod=20)
df_split["SMA50"] = talib.SMA(close_price, timeperiod=50)
df_split["SMA100"] = talib.SMA(close_price, timeperiod=100)
df_split["SMA200"] = talib.SMA(close_price, timeperiod=200)

# ボリンジャーバンド
df_split["upper_band"], df_split["middle_band"], df_split["lower_band"] = talib.BBANDS(close_price, timeperiod=20)

# MACD
df_split["macd"], df_split["macdsignal"], df_split["macdhist"] = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)

# MACDクロスを計算
df_split["macd_cross"] = np.where((df_split["macd"].shift(1) < df_split["macdsignal"].shift(1)) & (df_split["macd"] > df_split["macdsignal"]), 1, 0)
df_split["macd_cross"] = np.where((df_split["macd"].shift(1) > df_split["macdsignal"].shift(1)) & (df_split["macd"] < df_split["macdsignal"]), -1, df_split["macd_cross"])

# RSI
df_split["RSI"] = talib.RSI(close_price, timeperiod=14)

# ストキャスティクス
df_split["slowk"], df_split["slowd"] = talib.STOCH(high_price, low_price, close_price, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# ADX
df_split["ADX"] = talib.ADX(high_price, low_price, close_price, timeperiod=14)

# CCI
df_split["CCI"] = talib.CCI(high_price, low_price, close_price, timeperiod=14)

# ATR
df_split["ATR"] = talib.ATR(high_price, low_price, close_price, timeperiod=14)

# ROC
df_split["ROC"] = talib.ROC(close_price, timeperiod=10)

# Williams %R
df_split["Williams %R"] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)

# 値上がり率（単純収益率）を計算
df_split["return"] = df_split["close_price"].pct_change()

# 欠損値を削除
df_split.dropna(inplace=True)

# 学習データとテストデータに分割
train_size = 0.8  # 学習データの割合を指定
train_df = df_split[:int(len(df_split) * train_size)]
test_df = df_split[int(len(df_split) * train_size):]

print(f"全データ数: {len(df_split)}")
print(f"学習データ数: {len(train_df)}")
print(f"テストデータ数: {len(test_df)}")

# 多重共線性のチェック
def check_multicollinearity(data, threshold=5):
    features = data.columns.tolist()
    vif_scores = []
    for feature in features:
        vif_score = variance_inflation_factor(data[features].values, features.index(feature))
        vif_scores.append((feature, vif_score))
    multicollinearity_features = [feature for feature, vif_score in vif_scores if vif_score > threshold]
    return multicollinearity_features

# 多重共線性が認められる特徴量を削除
def remove_multicollinearity(data, threshold=5):
    multicollinearity_features = check_multicollinearity(data, threshold)
    data.drop(columns=multicollinearity_features, inplace=True)
    return data

# 学習データとテストデータから多重共線性が認められる特徴量を削除
train_df = remove_multicollinearity(train_df)
test_df = remove_multicollinearity(test_df)

print(f"多重共線性が認められる特徴量を削除後の学習データ数: {len(train_df)}")
print(f"多重共線性が認められる特徴量を削除後のテストデータ数: {len(test_df)}")

# CSVファイルに出力
train_df.to_csv("/root/src/src/crypto/procesed/btc_5min_technical_analysis_train.csv")
test_df.to_csv("/root/src/src/crypto/procesed/btc_5min_technical_analysis_test.csv")