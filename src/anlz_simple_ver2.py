import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('ProcessedData/cumpany.csv')

# 業種の一覧を取得
industries = df['業種'].unique()
print("業種の一覧:")
print(industries)

# 業種の割合を計算
industry_counts = df['業種'].value_counts()
industry_percentages = industry_counts / len(df) * 100
print("\n業種の割合:")
print(industry_percentages)

# 市場ごとの業種の割合を計算
market_industry_counts = df.groupby(['市場', '業種']).size().unstack()
market_industry_percentages = market_industry_counts.div(market_industry_counts.sum(axis=1), axis=0) * 100
print("\n市場ごとの業種の割合:")
print(market_industry_percentages)

# 市場の企業数の割合を計算
market_counts = df['市場'].value_counts()
market_percentages = market_counts / len(df) * 100
print("\n市場の企業数の割合:")
print(market_percentages)