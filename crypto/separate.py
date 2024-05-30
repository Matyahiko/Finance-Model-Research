import json
import pandas as pd

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

# 分割したデータを保存
output_file_name = 'raw_data/btc/BTC-JPY_5min_2021-2024_split.json'
df_split.to_json(output_file_name, orient='records')
print(f"分割したデータを {output_file_name} に保存しました。")

