import urllib3
from dotenv import dotenv_values
from datetime import datetime, timedelta
import time
import requests

# 環境変数からIDとPWを読み込む
config = dotenv_values(".devcontainer/.env")
id = config["ID"]
pw = config["PW"]

# urllib3のPoolManagerインスタンスを生成
http = urllib3.PoolManager()

# 基本認証ヘッダーを作成
headers = urllib3.util.make_headers(basic_auth=f"{id}:{pw}")

def save_stock_prices(start_date, end_date):
    """
    指定された期間内の全株価データを取得して保存する関数。
    start_date: 開始日（YYYYMMDD形式）
    end_date: 終了日（YYYYMMDD形式）
    """
    # 開始日と終了日をdatetimeオブジェクトに変換
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    # リクエストカウンターとタイマー開始時刻を初期化
    request_count = 0
    start_time = time.time()

    while current_date <= end_date:
        if request_count >= 6:
            # 現在の時刻と開始時刻の差を計算
            elapsed_time = time.time() - start_time
            if elapsed_time < 3600:
                # 一時間未満の場合は待機
                print(f"Request limit reached. Waiting for {(3600 - elapsed_time)} seconds.")
                time.sleep(3600 - elapsed_time)
            # カウンターとタイマーをリセット
            request_count = 0
            start_time = time.time()

        # 現在の日付をYYYYMMDD形式に変換
        date_str = current_date.strftime("%Y%m%d")
        url = f"https://csvex.com/kabu.plus/csv/japan-all-stock-prices-2/daily/japan-all-stock-prices-2_{date_str}.csv"

        # 指定されたURLからデータを取得
        response = requests.post(url, headers=headers)
        request_count += 1  # リクエストカウントを増やす

        # リクエストごとに15秒間スリープ
        time.sleep(15)

        if response.status_code == 404:
            print(f"No data available for {date_str}, skipping.")
        elif response.status_code == 200:
            with open(f"raw_data/japan-all-stock-prices/{date_str}.csv", "wb") as f:
                f.write(response.content)
            print(f"Saved: japan-all-stock-prices-{date_str}.csv")
        else:
            print(f"Failed to download data for {date_str}, error code: {response.status_code}")

        # 次の日に移動
        current_date += timedelta(days=1)

# モードを選択
#mode = input("Choose mode: (1) Download data for a specified period, (2) Download data for today: ")

mode = "2"

if mode == "1":
    # 開始日と終了日を指定
    start_date = input("Enter start date (YYYYMMDD): ")
    end_date = input("Enter end date (YYYYMMDD): ")
    # 指定された期間で株価データを取得
    save_stock_prices(start_date, end_date)
elif mode == "2":
    # 今日の日付を取得
    today = datetime.now().strftime("%Y%m%d")
    # 今日の日付で株価データを取得
    save_stock_prices(today, today)
else:
    print("Invalid mode selected.")