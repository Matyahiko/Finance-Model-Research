import urllib3
from dotenv import dotenv_values
from datetime import datetime
import time
import schedule
import requests

# 環境変数からIDとPWを読み込む
config = dotenv_values(".devcontainer/.env")
id = config["ID"]
pw = config["PW"]

# urllib3のPoolManagerインスタンスを生成
http = urllib3.PoolManager()

# 基本認証ヘッダーを作成
headers = urllib3.util.make_headers(basic_auth=f"{id}:{pw}")

def save_today_stock_prices():
    """
    当日の株価データを取得して保存する関数。
    """
    # 現在の日付と時刻を取得
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    print(f"Script executed on {current_date} at {current_time}")

    # 当日の日付をYYYYMMDD形式で取得
    today = now.strftime("%Y%m%d")

    url = f"https://csvex.com/kabu.plus/csv/japan-all-stock-prices-2/daily/japan-all-stock-prices-2_{today}.csv"

    # 指定されたURLからデータを取得
    response = requests.post(url, headers=headers)

    if response.status_code == 404:
        print(f"No data available for {today}, skipping.")
    elif response.status_code == 200:
        with open(f"raw_data/japan-all-stock-prices/{today}.csv", "wb") as f:
            f.write(response.content)
        print(f"Saved: japan-all-stock-prices-{today}.csv")
    else:
        print(f"Failed to download data for {today}, error code: {response.status_code}")

# 毎日17時にsave_today_stock_prices関数を実行するようにスケジュールを設定
schedule.every().day.at("17:00").do(save_today_stock_prices)

while True:
    schedule.run_pending()
    time.sleep(1)