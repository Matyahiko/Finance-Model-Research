import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import japanize_matplotlib
from itertools import combinations

# CSVファイルを読み込む
file_path = "ProcessedData/stock_prices_summary.csv"
df = pd.read_csv(file_path, low_memory=False)

# 前日比(%)の列を取得
change_columns = [col for col in df.columns if col.startswith("前日比（％）_")]

# 前日比(%)の列を数値に変換 (例外処理を追加)
for col in change_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 前日比(%)の累積を計算
cumulative_change_columns = [f"累積前日比_{col[7:]}" for col in change_columns]
df[cumulative_change_columns] = df[change_columns].cumsum()

# 市場と業種でグループ化
grouped = df.groupby(["市場", "業種"])

industry_mean_corr = {}

for (market, industry), group in grouped:
    fig, ax = plt.subplots(figsize=(12, 8))

    # 業種内の銘柄の組み合わせを取得
    symbols = group["SC"].unique()
    symbol_pairs = list(combinations(symbols, 2))

    # 業種内の平均ペアワイズ相関係数を計算
    corr_sum = 0
    for pair in symbol_pairs:
        symbol1, symbol2 = pair
        data1 = group[group["SC"] == symbol1][change_columns].values[0]
        data2 = group[group["SC"] == symbol2][change_columns].values[0]
        corr = np.corrcoef(data1, data2)[0, 1]
        corr_sum += corr

    mean_corr = corr_sum / len(symbol_pairs)
    industry_mean_corr[(market, industry)] = mean_corr

    # 前日比(%)の累積をプロット
    for _, row in group.iterrows():
        cumulative_changes = row[cumulative_change_columns].tolist()
        ax.plot(cumulative_change_columns, cumulative_changes)

    ax.set_title(f"市場: {market}, 業種: {industry}")
    ax.set_xlabel("日付")
    ax.set_ylabel("前日比(%)の累積")

    # 平均ペアワイズ相関係数をグラフに追加
    ax.text(0.95, 0.05, f"平均ペアワイズ相関係数: {mean_corr:.2f}", transform=ax.transAxes, ha="right", va="bottom")

    # 日付を適度な間隔で表示
    num_dates = len(cumulative_change_columns)
    if num_dates <= 10:
        date_indices = range(num_dates)
    else:
        date_indices = range(0, num_dates, num_dates // 10)
    ax.set_xticks(date_indices)
    ax.set_xticklabels([cumulative_change_columns[i] for i in date_indices], rotation=45, ha="right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"fig/cumulative_change_{market}_{industry}.png")
    plt.close()

print("プロットが完了しました。")

# 平均ペアワイズ相関係数を順位付けして表示
sorted_industry_mean_corr = sorted(industry_mean_corr.items(), key=lambda x: x[1], reverse=True)
print("業種ごとの平均ペアワイズ相関係数 (降順):")
for (market, industry), mean_corr in sorted_industry_mean_corr:
    print(f"市場: {market}, 業種: {industry}, 平均ペアワイズ相関係数: {mean_corr:.2f}")