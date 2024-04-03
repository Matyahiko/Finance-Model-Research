from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import datetime

def get_stock_data(symbol, min_range, min_interval):
    try:
        # 指定した銘柄の分足データ取得します
        # 日本株の場合は　末尾に".T"が必要です
        # 現在は".T"入ってますので、日本株は気にせず使えます。
        stock_data = share.Share(f"{symbol}.T")

        # 指定した銘柄コードでデータを取得しにいきます。
        data = stock_data.get_historical(share.PERIOD_TYPE_DAY, min_range, share.FREQUENCY_TYPE_MINUTE, min_interval)

        # 加工できる形にデータ形式を変更（dataframeに変更です。
        df = pd.DataFrame(data)

        # このままだと時間が使えないので、日本時間に変換します
        # timestampを日本時間へ変換
        df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")
        df["datetime_jst"] = df["datetime"] + datetime.timedelta(hours=9)
        df["datetime"] = df["datetime"].astype(str)

        # datetime_jst に格納されるデータが日本時間になります。
        df["datetime_jst"] = df["datetime_jst"].astype(str)

        # データ取得できてるか確認用に末尾五行を表示
        print(df.tail(5))

        # 加工のために元データをコピーし、列名を変更し、データ抜けを削除した後、今回使うデータだけにしています
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

        # 確認用
        print(processed_data.tail(5))

        # processed_data.to_csv(f"{symbol}_{min_interval}min.csv", index=False)

        return processed_data

    except YahooFinanceError as e:
        print(f"Error fetching data for symbol {symbol}: {e}")
        return None

if __name__ == "__main__":
    # 指定した銘柄コードと期間でデータを取得をします。
    # symbols → 個別株・指数のコード
    symbol = "1379"

    # _minRange_ → 取得期間　（5分足は60日まで、1分足は7日まで取得できます)
    # 取得期間を変えたい場合は　60 を変更してください
    min_range = 61

    # _min → 　分足の指定　５だと5分足　１だと1分足です
    min_interval = 5

    stock_data = get_stock_data(symbol, min_range, min_interval)