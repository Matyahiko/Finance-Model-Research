import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib



# TSVファイルを読み込む
df = pd.read_csv('/root/src/research/rakuten_data/travel_aspect_sentiment/travel_aspect_sentiment.tsv', sep='\t')

# ヒストグラムを表示したい列を選択
columns = ['朝食', '夕食', '風呂', 'サービス', '立地', '設備・アメニティ', '部屋']

# 各列のデータ型を数値型に変換
for column in columns:
    df[column + '_ポジティブ'] = pd.to_numeric(df[column + '_ポジティブ'], errors='coerce')
    df[column + '_ネガティブ'] = pd.to_numeric(df[column + '_ネガティブ'], errors='coerce')

# 欠損値を0で埋める
df = df.fillna(0)

# 各要素の頻度を計算
freq_data = {}
for column in columns:
    freq_data[column] = {
        'ポジティブ': df[column + '_ポジティブ'].sum(),
        'ネガティブ': df[column + '_ネガティブ'].sum()
    }

# 頻度データをデータフレームに変換
freq_df = pd.DataFrame(freq_data).T

# プロットを作成
fig, ax = plt.subplots(figsize=(12, 8))

# 頻度データをプロット
freq_df.plot(kind='bar', ax=ax)

# グラフのタイトルと軸ラベルを設定
ax.set_title('Frequency of Review Scores')
ax.set_xlabel('Element')
ax.set_ylabel('Frequency')

# 凡例を表示
ax.legend(title='Sentiment')

# グラフを表示
plt.tight_layout()
plt.savefig('/root/src/research/fig/frequency_of_review_scores.png')