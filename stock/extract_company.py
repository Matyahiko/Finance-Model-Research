import csv
import json

# 入力CSVファイルのパス
input_file = "/root/src/raw_data/japan-all-stock-prices/20230301.csv"

# 出力CSVファイルのパス
output_csv_file = "ProcessedData/cumpany.csv"

# 出力JSONファイルのパス
output_json_file = "ProcessedData/cumpany.json"

# 抽出する列のインデックス
columns_to_extract = [0, 1, 2, 3]  # "SC", "名称", "市場", "業種"のインデックス

# CSVファイルから指定列を抽出してリストに格納
data = []
with open(input_file, "r", encoding="shift-jis") as file:
    reader = csv.reader(file)
    for row in reader:
        extracted_row = [row[i] for i in columns_to_extract]
        data.append(extracted_row)

# CSVファイルに書き込む
with open(output_csv_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# JSONファイルに書き込む
with open(output_json_file, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("データの抽出と保存が完了しました。")