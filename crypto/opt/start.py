import pandas as pd
pd.options.display.float_format = '{:.2f}'.format

import requests
import time
import sys
import traceback
from datetime import datetime
from bb_api import BbApi
from slack_notify import SlackNotify

# CryptoCompareからBTC/JPYのヒストリカルデータを取得しローソク足を生成
def get_candles(timeframe, limit):
    base_url = f"https://min-api.cryptocompare.com/data/histo{timeframe}"
    
    params = {
        "fsym": "BTC",
        "tsym": "JPY",
        "limit": limit
    }
    
    res = requests.get(base_url, params, timeout = 10).json()
    
    time, open, high, low, close = [], [], [], [], []
    
    for i in res["Data"]:
        time.append(datetime.fromtimestamp(i["time"]))
        open.append(i["open"])
        high.append(i["high"])
        low.append(i["low"])
        close.append(i["close"])
        
    candles = pd.DataFrame({
        "Time": time,
        "Open": open,
        "High": high,
        "Low": low,
        "Close": close
        }
    )
    
    return candles

# 単純移動平均線を算出
def make_sma(candles, span):
    return pd.Series(candles["Close"]).rolling(window = span).mean()

bb_api = BbApi()
symbol = "BTC/USD"
amount = 1

# slack_notify = SlackNotify()
# slack_notify.send("Start trading")

print("Start trading")

#Botの起動
while True:
    try:
        candles = get_candles("minute", 1000).set_index("Time")
        sma_5 = make_sma(candles, 5)
        sma_13 =  make_sma(candles, 13)
        
       # 短期移動平均線 > 長期移動平均線 の状態が3本続いたらゴールデンクロス（騙し防止のために判断まで少し待つ）
        golden_cross = sma_5.iloc[-1] > sma_13.iloc[-1] \
            and sma_5.iloc[-2] > sma_13.iloc[-2] \
            and sma_5.iloc[-3] > sma_13.iloc[-3] \
            and sma_5.iloc[-4] < sma_13.iloc[-4]

        # 短期移動平均線 < 長期移動平均線 の状態が3本続いたらデッドクロス（騙し防止のために判断まで少し待つ）
        dead_cross = sma_5.iloc[-1] < sma_13.iloc[-1] \
            and sma_5.iloc[-2] < sma_13.iloc[-2] \
            and sma_5.iloc[-3] < sma_13.iloc[-3] \
            and sma_5.iloc[-4] > sma_13.iloc[-4]
            
        position = bb_api.get_position(symbol)
        
        if position["side"] == "None":
            if golden_cross:
                order = bb_api.create_order(symbol, "market", "buy", amount)
                price = order["price"]
                #slack_notify.send(f"Sell on the market: {price}")
                print(f"Buy on the market: {price}")
            elif dead_cross: # ノーポジかつデッドクロスが現れたら新規売り
                order = bb_api.create_order(symbol, "market", "sell", amount)
                price = order["price"]
                # slack_notify.send(f"Sell on the market {price}")
                print(f"Sell on the market: {price}")

        elif position["side"] == "Buy" and dead_cross: # 買いポジション保有中かつデッドクロスが現れたらドテン売り
            order = bb_api.create_order(symbol, "market", "sell", amount * 2)
            price = order["price"]
            #slack_notify.send(f"Stop and sell reversely {price}")
            print(f"Stop and sell reversely {price}")

        elif position["side"] == "Sell" and golden_cross: # 売りポジション保有中かつゴールデンクロスが現れたらドテン買い
            order = bb_api.create_order(symbol, "market", "buy", amount * 2)
            price = order["price"]
            #slack_notify.send(f"Stop and buy reversely {price}")
            print(f"Stop and buy reversely {price}")

        time.sleep(30)
    except:
        # slack_notify.send(traceback.format_exc())
        print(traceback.format_exc())
        sys.exit() # 何か例外が生じた際はLINE通知を飛ばしてBotを停止する