from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

class StockDataFetcher:
    def __init__(self, symbol, min_range, min_interval):
        self.symbol = symbol
        self.min_range = min_range
        self.min_interval = min_interval

    def fetch_data(self):
        try:
            stock_data = share.Share(f"{self.symbol}.T")
            data = stock_data.get_historical(share.PERIOD_TYPE_DAY, self.min_range, share.FREQUENCY_TYPE_MINUTE, self.min_interval)
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")
            df["datetime_jst"] = df["datetime"] + datetime.timedelta(hours=9)
            df["datetime"] = df["datetime"].astype(str)
            df["datetime_jst"] = df["datetime_jst"].astype(str)
            print(df.tail(5))

            df_copy = df.copy()
            df_copy.rename(columns={
                'datetime_jst': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            df_copy = df_copy.dropna()
            processed_data = df_copy.reindex(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            print(processed_data.tail(5))
            return processed_data
        except YahooFinanceError as e:
            print(f"Error fetching data for symbol {self.symbol}: {e}")
            return None

def parallel_fetch_data(symbols, min_range, min_interval, output_dir, max_workers=10):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for symbol in symbols:
            fetcher = StockDataFetcher(symbol, min_range, min_interval)
            future = executor.submit(fetcher.fetch_data)
            futures.append((symbol, future))
            time.sleep(1)  # リクエストを1秒ごとに送信

        for symbol, future in futures:
            result = future.result()
            results[symbol] = result
            if result is not None:
                output_path = os.path.join(output_dir, f"{symbol}_{min_interval}min.csv")
                result.to_csv(output_path, index=False)
    return results

if __name__ == "__main__":
    csv_file = "ProcessedData/cumpany.csv"  # 銘柄コードが含まれるCSVファイル
    output_dir = "/root/src/raw_data/japan-all-stock-prices_yf2/"  # 出力ディレクトリ
    min_range = 50
    min_interval = 15

    debug_mode = False  # デバッグモードのフラグ

    if debug_mode:
        symbols = ["1379", "9984", "6758"]  # デバッグ用の銘柄コードを指定
    else:
        df = pd.read_csv(csv_file)
        symbols = df["SC"].tolist()  # SC列から銘柄コードを取得

    os.makedirs(output_dir, exist_ok=True)  # 出力ディレクトリが存在しない場合は作成

    stock_data = parallel_fetch_data(symbols, min_range, min_interval, output_dir, max_workers=10)