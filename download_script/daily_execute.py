import os
import time
import schedule

def run_script(script_name):
    print(f"実行中: {script_name}")
    os.system(f"python {script_name}")

# 実行するスクリプトと時間を指定
scripts_to_run = [
    {"name": "/root/src/download_script/stock_download.py", "time": "4:00"},
    {"name": "/root/src/download_script/tdnet_download.py", "time": "3:00"},
    {"name": "/root/src/download_script/news_download.py", "time": "2:00"}
]

# スケジュールを設定
for script in scripts_to_run:
    schedule.every().day.at(script["time"]).do(run_script, script["name"])

# メインループ
while True:
    schedule.run_pending()
    time.sleep(60)