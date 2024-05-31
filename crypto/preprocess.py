import json
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

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

print(f"全データ数: {df_split.shape}")
print(f"学習データ数: {train_df.shape}")
print(f"テストデータ数: {test_df.shape}")

def check_multicollinearity(train_data, test_data, threshold=0.7):
    # トレインデータの相関係数行列を計算
    corr_matrix = train_data.corr()

    # 上三角行列を取得
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 閾値を超える相関係数を持つ説明変数のペアを特定
    multicollinearity_pairs = []
    columns_to_drop = []
    for i in range(len(upper_tri.columns)):
        for j in range(i+1, len(upper_tri.columns)):
            if abs(upper_tri.iloc[i, j]) > threshold:
                multicollinearity_pairs.append((upper_tri.index[i], upper_tri.columns[j]))
                columns_to_drop.append(upper_tri.columns[j])  # 削除する列を追加

    # 重複する列名を削除
    columns_to_drop = list(set(columns_to_drop))

    # トレインデータとテストデータから特徴量の列を削除
    train_data_cleaned = train_data.drop(columns=columns_to_drop)
    test_data_cleaned = test_data.drop(columns=columns_to_drop)

    # ヒートマップを作成
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()

    # ヒートマップを保存
    plt.savefig('crypto/fig/correlation_heatmap.png')
    plt.close()

    return train_data_cleaned, test_data_cleaned

# トレインデータのみで共線性チェックを行い、同じ特徴量をトレインデータとテストデータから削除
train_df_cleaned, test_df_cleaned = check_multicollinearity(train_df, test_df)

print(f"多重共線性が認められる特徴量を削除後の学習データ数: {train_df_cleaned.shape}")
print(f"多重共線性が認められる特徴量を削除後のテストデータ数: {test_df_cleaned.shape}")


# CSVファイルに出力
train_df.to_csv("crypto/procesed/btc_5min_technical_analysis_train.csv")
test_df.to_csv("crypto/procesed/btc_5min_technical_analysis_test.csv")