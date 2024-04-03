from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def parallel_fetch_data(symbols, min_range, min_interval):
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for symbol in symbols:
            fetcher = StockDataFetcher(symbol, min_range, min_interval)
            future = executor.submit(fetcher.fetch_data)
            futures.append((symbol, future))

        for symbol, future in futures:
            result = future.result()
            results[symbol] = result

    return results

if __name__ == "__main__":
    symbols = ["1379", "9984", "6758"]  # 複数の銘柄コードを指定
    min_range = 3
    min_interval = 5

    stock_data = parallel_fetch_data(symbols, min_range, min_interval)

    for symbol, data in stock_data.items():
        if data is not None:
            data.to_csv(f"{symbol}_{min_interval}min.csv", index=False)