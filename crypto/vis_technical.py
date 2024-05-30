import json
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import os
from matplotlib.backends.backend_pdf import PdfPages

#csvファイルを読み込む
df = pd.read_csv('src/crypto/procesed/btc_15min_technical_analysis_train.csv')

df.set_index('close_time', inplace=True)


# 各指標を可視化
#indicators = ['SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA200', 'upper_band', 'middle_band', 'lower_band','macd',"macd_cross", 'macdhist','macdsignal', 'RSI', 'slowk', 'slowd', 'ADX', 'CCI', 'ATR', 'ROC', 'Williams %R',"return"]
indicators = ["return"]

for indicator in indicators:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[indicator])
    plt.title(indicator)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f'src/crypto/fig/{indicator}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(df[indicator], bins=50, edgecolor='black')
    plt.title(f"{indicator} - Histogram")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'src/crypto/fig/{indicator}_histogram.png')
    plt.close()

print("各指標の要約統計量:")
for indicator in indicators:
    print(f"\n{indicator}:")
    print(f"最小値: {df[indicator].min():.4f}")
    print(f"最大値: {df[indicator].max():.4f}")
    print(f"平均値: {df[indicator].mean():.4f}")
    print(f"中央値: {df[indicator].median():.4f}")
    print(f"標準偏差: {df[indicator].std():.4f}")
    print(f"尖度: {kurtosis(df[indicator]):.4f}")
    print(f"歪度: {skew(df[indicator]):.4f}")

# # A4サイズのPDFファイルを作成
# with PdfPages('technical_indicators.pdf') as pdf:
#     for i in range(0, len(indicators), 6):
#         fig, axs = plt.subplots(2, 3, figsize=(11.69, 8.27), tight_layout=True)
#         axs = axs.ravel()

#         for j in range(6):
#             if i + j < len(indicators):
#                 axs[j].plot(df.index, df[indicators[i + j]])
#                 axs[j].set_title(indicators[i + j])
#                 axs[j].set_xlabel('Date')
#                 axs[j].set_ylabel('Value')
#                 axs[j].grid(True)
#             else:
#                 fig.delaxes(axs[j])

#         pdf.savefig(fig)
#         plt.close()