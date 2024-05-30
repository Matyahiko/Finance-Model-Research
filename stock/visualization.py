import pandas as pd
import mplfinance as mpf

# CSVファイルを読み込む。出来高列のデータタイプ指定を削除（後で処理）
df = pd.read_csv("ProcessedData/japan-all-stock-prices_filtered_7203.csv")

# 必要な列だけを選択し、カラム名を英語に変更する
columns = ["日付", "始値", "高値", "安値", "株価", "出来高"]
new_column_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
df = df[columns]
df.columns = new_column_names

# "-" を含む数値列をNaNに置き換える
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 日付をdatetime型に変換し、インデックスとして設定する
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df = df.sort_values('Date')
print(df['Date'])
daily = df.set_index('Date')


daily.index.name = "Date"

# mplfinanceを使用してプロットし、画像として保存する
mpf.plot(daily, style="charles", 
         savefig="fig/stock_prices_plot.png",mav=(3,6,9),volume=True)

mpf.plot(daily, style="charles", savefig=dict(fname="fig/stock_prices_plot.svg", format="svg"),mav=(3,6,9),volume=True)

print("Plot saved as 'stock_prices_plot.png'.")
