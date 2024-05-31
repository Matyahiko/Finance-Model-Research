import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from itertools import combinations

# CSVファイルを読み込む
file_path = "ProcessedData/stock_prices_summary.csv"
dataframe = pd.read_csv(file_path, low_memory=False)

# 前日比(%)の列を取得
change_columns = [col for col in dataframe.columns if col.startswith("前日比（％）_")]

# 前日比(%)の列を数値に変換 (例外処理を追加)
for col in change_columns:
    dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

# 前日比(%)の累積を計算
cumulative_change_columns = [f"累積前日比_{col[7:]}" for col in change_columns]
dataframe[cumulative_change_columns] = dataframe[change_columns].cumsum()

# 市場と業種でグループ化
grouped = dataframe.groupby(["市場", "業種"])
industry_mean_correlation = {}

for (market, industry), group in grouped:
    fig, ax = plt.subplots(figsize=(12, 8))

    # 業種内の銘柄の組み合わせを取得
    symbols = group["SC"].unique()
    symbol_pairs = list(combinations(symbols, 2))

    # 業種内の平均ペアワイズ相関係数を計算
    if len(symbol_pairs) > 0:
        correlation_sum = 0
        for pair in symbol_pairs:
            symbol1, symbol2 = pair
            data1 = group[group["SC"] == symbol1][change_columns].values[0]
            data2 = group[group["SC"] == symbol2][change_columns].values[0]
            correlation = np.corrcoef(data1, data2)[0, 1]
            correlation_sum += correlation
        mean_correlation = correlation_sum / len(symbol_pairs)
        industry_mean_correlation[(market, industry)] = mean_correlation
    else:
        mean_correlation = np.nan
        industry_mean_correlation[(market, industry)] = mean_correlation

    # 前日比(%)の累積をプロット
    for _, row in group.iterrows():
        cumulative_changes = row[cumulative_change_columns].tolist()
        ax.plot(cumulative_change_columns, cumulative_changes)

    ax.set_title(f"市場: {market}, 業種: {industry}")
    ax.set_xlabel("日付")
    ax.set_ylabel("前日比(%)の累積")

    # 平均ペアワイズ相関係数をグラフに追加
    if not np.isnan(mean_correlation):
        ax.text(0.95, 0.05, f"平均ペアワイズ相関係数: {mean_correlation:.2f}", transform=ax.transAxes, ha="right", va="bottom")
    else:
        ax.text(0.95, 0.05, "平均ペアワイズ相関係数: N/A", transform=ax.transAxes, ha="right", va="bottom")

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
sorted_industry_mean_correlation = sorted(industry_mean_correlation.items(), key=lambda x: x[1], reverse=True)
print("業種ごとの平均ペアワイズ相関係数 (降順):")
for (market, industry), mean_correlation in sorted_industry_mean_correlation:
    print(f"市場: {market}, 業種: {industry}, 平均ペアワイズ相関係数: {mean_correlation:.2f}")