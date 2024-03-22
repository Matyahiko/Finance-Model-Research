import pandas as pd
import glob
import os

def process_stock_data(folder_path, output_folder, stock_code=None, debug_mode=False):
    """
    指定されたフォルダ内の全てのCSVファイルを読み込み、結合したデータフレームを作成し、
    指定された銘柄コードでフィルタリングしたデータフレームを作成する。
    """
    # 指定フォルダ内の全てのCSVファイルをリストアップ
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 空のDataFrameを作成
    df_total = pd.DataFrame()

    # 各CSVファイルを読み込んで、df_totalに追加
    for file in csv_files:
        # Shift-JISでエンコードされたCSVファイルを読み込む、"SC"列を文字列として読み込む
        df = pd.read_csv(file, encoding="shift_jis", dtype={"SC": str})
        
        if debug_mode:
            print(f"Processing file: {file}")
            print(f"Data types:\n{df.dtypes}\n")
        
        # 読み込んだデータフレームをdf_totalに追加
        df_total = pd.concat([df_total, df], ignore_index=True)

    # "日付"列でdf_totalをソート
    df_total = df_total.sort_values(by="日付", ignore_index=True)

    # 全データをCSVファイルとして保存
    output_file = os.path.join(output_folder, "japan-all-stock-prices_combined.csv")
    df_total.to_csv(output_file, index=False, encoding="utf-8")

    if stock_code:
        # 指定された銘柄コードでデータをフィルタリング
        filtered_data = df_total[df_total["SC"] == stock_code]
        
        # フィルタリングされたデータをCSVファイルとして保存
        filtered_output_file = os.path.join(output_folder, f"japan-all-stock-prices_filtered_{stock_code}.csv")
        filtered_data.to_csv(filtered_output_file, index=False, encoding="utf-8")

    return df_total, filtered_data if stock_code else None

if __name__ == "__main__":
    folder_path = "japan-all-stock-prices/"
    output_folder = "ProcessedData"
    stock_code = "7203"
    debug_mode = True

    os.makedirs(output_folder, exist_ok=True)
    df_total, filtered_data = process_stock_data(folder_path, output_folder, stock_code, debug_mode)
    