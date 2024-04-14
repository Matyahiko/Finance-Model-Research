import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time


# Binanceの取引所インスタンスを作成
#ex = ccxt.binance()
ex = ccxt.coinbase()
ex.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

# 開始日時と終了日時を設定（3年前から現在まで）
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

# Unix timestampに変換
start_ts = int(start_date.timestamp() * 1000)
end_ts = int(end_date.timestamp() * 1000)

# データを格納するリスト
ohlcv_list = []

# APIリクエストを繰り返し、データを取得
while start_ts < end_ts:
    print(f"Fetching data from {pd.to_datetime(start_ts, unit='ms')}")
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '10m', since=start_ts, limit=10)
    ohlcv_list.extend(ohlcv)
    start_ts = ohlcv[-1][0]
    
    # API制限に引っかからないように1秒待機
    time.sleep(1)

# リストをデータフレームに変換
df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# timestampをdatetime形式に変換し、インデックスに設定
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# インデックスを昇順にソート
df.sort_index(inplace=True)

# 結果を表示
print(df.head())
print(df.tail())
print(len(df))