import pandas as pd
import glob
import os

def process_stock_data(folder_path, output_folder, industry_code, first_stock_code=None, debug_mode=False):
    """
    指定されたフォルダ内の全ての企業ごとのCSVファイルを読み込み、結合したデータフレームを作成し、
    日付を行方向、企業を列方向に配置したデータフレームを作成する。
    先頭の企業はfirst_stock_codeで指定できる。
    """
    # 指定フォルダ内の全ての企業ごとのCSVファイルをリストアップ
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 空のDataFrameを作成
    df_total = pd.DataFrame()

    # 各企業のCSVファイルを読み込んで、df_totalに追加
    for file in csv_files:
        # UTF-8でエンコードされたCSVファイルを読み込む
        df = pd.read_csv(file, encoding="utf-8", dtype={"SC": str})

        if debug_mode:
            print(f"Processing file: {file}")
            print(f"Data types:\n{df.dtypes}\n")

        # 読み込んだデータフレームをdf_totalに追加
        df_total = pd.concat([df_total, df], ignore_index=True)

    # 業種コードで絞り込み
    df_total = df_total[df_total["業種"] == industry_code]

    # "日付"列でdf_totalをソート
    df_total = df_total.sort_values(by="日付", ignore_index=True)

    # 日付を行方向、企業を列方向に配置したデータフレームを作成
    df_wide = df_total.pivot(index="日付", columns="SC")

    # 列名の先頭に"SC"列のコードを追加
    df_wide.columns = [f"{col[1]}_{col[0]}" for col in df_wide.columns]

    # 先頭の企業を指定されたSCコードの企業に変更
    if first_stock_code:
        columns = df_wide.columns.tolist()
        first_stock_columns = [col for col in columns if col.startswith(first_stock_code)]
        other_columns = [col for col in columns if not col.startswith(first_stock_code)]
        df_wide = df_wide[first_stock_columns + other_columns]

    # 全データをCSVファイルとして保存
    output_file = os.path.join(output_folder, f"japan-{industry_code}-stock-prices_wide.csv")
    df_wide.to_csv(output_file, index=True, encoding="utf-8")

    if debug_mode == True:
        print(f"Data types:\n{df_wide.dtypes}\n")
        print(f"Data shape: {df_wide.shape}\n")
        print(f"Data head:\n{df_wide.head()}\n")
        print(f"Data tail:\n{df_wide.tail()}\n")
        print(f"Data file: {output_file}\n")

    return df_wide

if __name__ == "__main__":
    folder_path = "add_feature/"
    output_folder = "ProcessedData"
    industry_code = "輸送用機器"
    first_stock_code = "7203"  # トヨタ自動車のSCコード
    debug_mode = False

    os.makedirs(output_folder, exist_ok=True)
    df_wide = process_stock_data(folder_path, output_folder, industry_code, first_stock_code, debug_mode)