import os
import pandas as pd

# 日毎の株価CSVファイルが格納されているフォルダのパス
input_folder = "raw_data/japan-all-stock-prices"

# 出力先のフォルダのパス
output_folder = "add_feature"

# 移動平均の日数
moving_averages = [15, 21, 25]

# フォルダ内のCSVファイルを取得
csv_files = [file for file in os.listdir(input_folder) if file.endswith(".csv")]

# 企業ごとのデータフレームを格納する辞書
company_data = {}

# 各CSVファイルを読み込む
for file in csv_files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path, encoding="shift-jis")
    
    # 企業コードごとにデータを分割
    for sc, group in df.groupby("SC"):
        if sc not in company_data:
            company_data[sc] = group
        else:
            company_data[sc] = pd.concat([company_data[sc], group])

# 各企業のデータに移動平均の列を追加
for sc, df in company_data.items():
    df = df.sort_values("日付")  # 日付で昇順にソート
    
    # "株価"列を数値に変換し、非数値を NaN に置き換える
    df["株価"] = pd.to_numeric(df["株価"], errors="coerce")
    
    for ma in moving_averages:
        column_name = f"移動平均{ma}日"
        if df["株価"].notnull().all():
            df[column_name] = df["株価"].rolling(window=ma).mean()
        else:
            print(f"警告: {sc}の株価データに非数値が含まれています。")
    
    # 出力先のファイルパスを作成
    output_file = os.path.join(output_folder, f"{sc}.csv")
    
    # CSVファイルに保存 (UTF-8エンコーディング)
    df.to_csv(output_file, index=False, encoding="utf-8")

print("処理が完了しました。")