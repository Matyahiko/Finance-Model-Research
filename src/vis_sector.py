import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import japanize_matplotlib


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

# 業種ごとにグループ化してプロット
grouped = df.groupby("業種")

for industry, group in grouped:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 前日比(%)の累積をプロット
    for _, row in group.iterrows():
        cumulative_changes = row[cumulative_change_columns].tolist()
        ax.plot(cumulative_change_columns, cumulative_changes, label=row["名称"])

    ax.set_title(f"業種: {industry}")
    ax.set_xlabel("日付")
    ax.set_ylabel("前日比(%)の累積")
    ax.set_xticks(range(len(cumulative_change_columns)))
    ax.set_xticklabels(cumulative_change_columns, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"fig/cumulative_change_{industry}.png")
    plt.close()

    
    
print("プロットが完了しました。")