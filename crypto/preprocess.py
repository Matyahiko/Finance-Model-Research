import json
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_distribution(df, feature):
    """
    データフレームの特定の列の分布を可視化する関数

    Args:
        df (pandas.DataFrame): 分布を可視化するデータフレーム
        feature (str): 分布を可視化する列名
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df[feature], bins=30, edgecolor='black')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"crypto/fig/{feature}_distribution.png")  

def remove_multicollinearity(train_data, test_data, threshold=0.7):
    """
    多重共線性を示す特徴量を削除する関数

    Args:
        train_data (pandas.DataFrame): 学習データのデータフレーム
        test_data (pandas.DataFrame): テストデータのデータフレーム
        threshold (float, optional): 多重共線性の閾値. Defaults to 0.7.

    Returns:
        tuple: 多重共線性が認められる特徴量を削除後の学習データとテストデータのタプル
    """
    corr_matrix = train_data.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    columns_to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > threshold)]

    train_data_cleaned = train_data.drop(columns=columns_to_drop)
    test_data_cleaned = test_data.drop(columns=columns_to_drop)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('crypto/fig/correlation_heatmap.png')
    plt.close()

    return train_data_cleaned, test_data_cleaned

def calculate_target_variable(df, short_period=5, long_period=20, rsi_buy_threshold=30, rsi_sell_threshold=70, profit_threshold=0.005):
    """
    各時間ステップでのエントリールールとその利益率を計算する関数

    Args:
        df (pandas.DataFrame): 計算対象のデータフレーム
        short_period (int, optional): 短期移動平均線の期間. Defaults to 5.
        long_period (int, optional): 長期移動平均線の期間. Defaults to 20.
        rsi_buy_threshold (int, optional): RSIの買いシグナルの閾値. Defaults to 30.
        rsi_sell_threshold (int, optional): RSIの売りシグナルの閾値. Defaults to 70.
        profit_threshold (float, optional): 利益率の閾値. Defaults to 0.005.

    Returns:
        pandas.DataFrame: エントリールールとその利益率を追加したデータフレーム
    """
    df["short_mavg"] = talib.SMA(df["close_price"], timeperiod=short_period)
    df["long_mavg"] = talib.SMA(df["close_price"], timeperiod=long_period)
    df["ma_cross"] = np.where((df["short_mavg"].shift(1) < df["long_mavg"].shift(1)) & (df["short_mavg"] > df["long_mavg"]), 1,
                              np.where((df["short_mavg"].shift(1) > df["long_mavg"].shift(1)) & (df["short_mavg"] < df["long_mavg"]), -1, 0))

    df["rsi_bb_signal"] = np.where((df["RSI"] < rsi_buy_threshold) & (df["close_price"] < df["lower_band"]), 1,
                                   np.where((df["RSI"] > rsi_sell_threshold) & (df["close_price"] > df["upper_band"]), -1, 0))

    df["macd_cross"] = np.where((df["macd"].shift(1) < df["macdsignal"].shift(1)) & (df["macd"] > df["macdsignal"]), 1,
                                np.where((df["macd"].shift(1) > df["macdsignal"].shift(1)) & (df["macd"] < df["macdsignal"]), -1, 0))

    df["return"] = df["close_price"].pct_change()
    df["target"] = np.where((df["ma_cross"] == 1) | (df["rsi_bb_signal"] == 1) | (df["macd_cross"] == 1), 1, 0)
    df["target"] = np.where((df["ma_cross"] == -1) | (df["rsi_bb_signal"] == -1) | (df["macd_cross"] == -1), -1, df["target"])
    df["target"] = np.where((df["target"] == 1) & (df["return"] >= profit_threshold), 1,
                            np.where((df["target"] == -1) & (df["return"] < 0), 1, df["target"]))

    return df

# JSONファイルからデータを読み込む
with open('RawData/btc/BTC-JPY_15min_2021-2024.json', 'r') as f:
    data = json.load(f)

# データフレームに変換
df = pd.DataFrame(data)

# 日時データをdatetime型に変換
df['close_time'] = pd.to_datetime(df['close_time'], format='%Y/%m/%d %H:%M:%S')

# データを分割
split_point = 1000  
df_split = df[:split_point]
print(f"全データ数: {len(df)}")
print(f"分割後のデータ数: {len(df_split)}")

# 日時データをインデックスに設定
df_split["close_time"] = pd.to_datetime(df_split["close_time"], unit="ms")
df_split.set_index("close_time", inplace=True)

# 各価格データを取得
open_price, high_price, low_price, close_price = df_split["open_price"], df_split["high_price"], df_split["low_price"], df_split["close_price"]

# 特徴量の計算
df_split["SMA5"], df_split["SMA10"], df_split["SMA20"] = talib.SMA(close_price, 5), talib.SMA(close_price, 10), talib.SMA(close_price, 20)
df_split["SMA50"], df_split["SMA100"], df_split["SMA200"] = talib.SMA(close_price, 50), talib.SMA(close_price, 100), talib.SMA(close_price, 200)
df_split["upper_band"], df_split["middle_band"], df_split["lower_band"] = talib.BBANDS(close_price, timeperiod=20)
df_split["macd"], df_split["macdsignal"], df_split["macdhist"] = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
df_split["RSI"] = talib.RSI(close_price, timeperiod=14)
df_split["slowk"], df_split["slowd"] = talib.STOCH(high_price, low_price, close_price, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df_split["ADX"] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
df_split["CCI"] = talib.CCI(high_price, low_price, close_price, timeperiod=14)
df_split["ATR"] = talib.ATR(high_price, low_price, close_price, timeperiod=14)
df_split["ROC"] = talib.ROC(close_price, timeperiod=10)
df_split["Williams %R"] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)

# エントリールールとその利益率の計算
df_split = calculate_target_variable(df_split)

# データの分布を可視化
visualize_distribution(df_split, "return")
visualize_distribution(df_split, "target")

# 不要な列を削除
df_split.drop("return", axis=1, inplace=True)
df_split.dropna(inplace=True)

# 学習データとテストデータに分割
train_size = 0.8
train_df = df_split[:int(len(df_split) * train_size)]
test_df = df_split[int(len(df_split) * train_size):]

print(f"全データ数: {df_split.shape}")
print(f"学習データ数: {train_df.shape}")
print(f"テストデータ数: {test_df.shape}")

# 多重共線性が認められる特徴量を削除
train_df_cleaned, test_df_cleaned = remove_multicollinearity(train_df, test_df)

print(f"多重共線性が認められる特徴量を削除後の学習データ数: {train_df_cleaned.shape}")
print(f"多重共線性が認められる特徴量を削除後のテストデータ数: {test_df_cleaned.shape}")
print(f"削除された特徴量: {set(train_df.columns) - set(train_df_cleaned.columns)}")

# 学習データとテストデータを保存
train_df_cleaned.to_csv("RawData/btc/btc_15min_technical_analysis_train.csv")
test_df_cleaned.to_csv("RawData/btc/btc_15min_technical_analysis_test.csv")