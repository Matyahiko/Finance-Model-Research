import pandas as pd
import numpy as np
import os
import time

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.models import Sequential

# CSVファイルを読み込む
df = pd.read_csv("/root/src/src/crypto/procesed/btc_15min_technical_analysis_train.csv", index_col="close_time", parse_dates=True)
df["macd_cross"] = df["macd_cross"].map({-1: 0, 0: 1, 1: 2})

# 特徴量と目的変数を定義
features = ["open_price", "high_price", "low_price", "close_price", "SMA5", "SMA10", "SMA20", "SMA50", "SMA100", "SMA200", "upper_band", "middle_band", "lower_band", "macd", "macdsignal","macd_cross", "macdhist", "RSI", "slowk", "slowd", "ADX", "CCI", "ATR", "ROC", "Williams %R"]
target = 'return'

# ハイパーパラメータの設定
input_dim = 5
timesteps = 10
lstm_units = 64

# ダミーデータの作成（実際には数値データを使用）
x_train = np.random.rand(100, timesteps, input_dim)
y_train = np.random.randint(2, size=(100, 1))

# モデルの構築
model = Sequential([
    LSTM(lstm_units, input_shape=(timesteps, input_dim)),
    Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルの概要を表示
model.summary()

# モデルの訓練
model.fit(x_train, y_train, epochs=5, batch_size=32)