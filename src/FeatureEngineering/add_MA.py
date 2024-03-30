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
    df = pd.read_csv(file_path, encoding="shift_jis")
    
    # 企業コードごとにデータを分割
    for sc, group in df.groupby("SC"):
        if sc not in company_data:
            company_data[sc] = group
        else:
            company_data[sc] = pd.concat([company_data[sc], group])

# 各企業のデータに移動平均の列を追加
for sc, df in company_data.items():
    df = df.sort_values("日付")  # 日付で昇順にソート
    
    for ma in moving_averages:
        column_name = f"移動平均{ma}日"
        df[column_name] = df["株価"].rolling(window=ma).mean()
    
    # 出力先のファイルパスを作成
    output_file = os.path.join(output_folder, f"{sc}.csv")
    
    # CSVファイルに保存
    df.to_csv(output_file, index=False, encoding="shift_jis")

print("処理が完了しました。")