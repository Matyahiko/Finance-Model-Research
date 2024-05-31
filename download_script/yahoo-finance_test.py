from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor
import time

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
            print(f"Symbol: {self.symbol}, Date Range: {self.min_range} days")
            print(f"First Date: {df['datetime_jst'].iloc[0]}, Last Date: {df['datetime_jst'].iloc[-1]}")
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
    symbols = ["1379"]  # 複数の銘柄コードを指定
    min_interval = 5
    min_days = 55  # 最小日数を指定
    max_days = 360  # 最大日数を指定
    
for days in range(min_days, max_days + 1):
    print(f"\nTesting with {days} days:")
    stock_data = parallel_fetch_data(symbols, days, min_interval)
    print(stock_data.shape)
    for symbol, data in stock_data.items():
        if data is not None:
            first_date = data['Date'].iloc[0]
            last_date = data['Date'].iloc[-1]
            date_range = pd.date_range(start=first_date, end=last_date, freq='D')
            if len(date_range) == days:
                print(f"Date range for {symbol} matches the requested {days} days.")
            else:
                print(f"Date range for {symbol} does not match the requested {days} days.")
                break  # ここにbreakを移動
        else:
            print(f"No data retrieved for {symbol} with {days} days.")
            break  # データが取得できなかった場合もbreakを追加
    else:
        continue  # すべての銘柄でデータが取得できた場合は次の日数に進む
    break  # いずれかの銘柄でデータが取得できなかった場合はループを終了            
                    
    time.sleep(5)  # 待機