import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta


logging.basicConfig(filename="/root/src/log/yahoo_finance_download.log", level=logging.INFO)

def finance_download(ticker,start_date,end_date,interval,proxy):
    # ticker → 個別株・指数のコード
    # start_date → 取得したい初めの日付
    # end_date  → 取得したい終わりの日付
    # interval  → 取得したい間隔（分足(1m)、日足(1d)、週足(1wk)、月足(1mo)）
    # proxy     → プロキシサーバーの設定 (無視体の場合は空白でも可)

    # 開始日と終了日をdatetimeオブジェクトに変換
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days = 1)

    # プロキシサーバーの設定
    if proxy == '':
        proxy = None    # プロキシサーバーが設定されていない場合
    #logging.info(f"Requesting data for {ticker} from {current_date} to {end_date} with interval {interval}")
    print(f"Requesting data for {ticker} from {current_date} to {end_date} with interval {interval}")

    try:
        # Yahoo Financeからデータを取得
        #logging.info(f"Downloading {ticker} data from Yahoo Finance...")
        print(f"Downloading {ticker} data from Yahoo Finance...")
        yf.pdr_override()
        df = yf.download(ticker, current_date, end_date, interval=interval , proxy=proxy)

        logging.info(f"Successful download.")
        print(f"Successful download.")
        return df
    except Exception as e:
        #logging.error(f"Error: {e}")
        print(f"Error: {ticker}")



def main():
    #トライ回数
    max_retries = 5

    for i in range(max_retries):
        df = finance_download('AAPL','20220101','20220131','1wk','')
        if df.empty:
            #logging.warning(f"Failed to download data. Retrying... ({i+1}/{max_retries})")
            print(f"Failed to download data. Retrying... ({i+1}/{max_retries})")
        else:
            print(df)
            break



if __name__ == "__main__":
    main()