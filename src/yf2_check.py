import pandas as pd
import os

dir = "/root/src/raw_data/japan-all-stock-prices_yf2/"
df = pd.read_csv("ProcessedData/cumpany.csv")

symbols = df["SC"].tolist()
failes = os.listdir(dir)
failes = [f.split("_")[0] for f in failes] 
symbols = [s for s in symbols if s not in failes]
print(len(symbols))

