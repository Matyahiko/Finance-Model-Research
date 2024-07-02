import pandas as pd

def extract_doc_ids(file_path, target_sec_code):
    # CSVファイルを読み込む
    try:
        df = pd.read_csv(file_path,sep="\t",encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty")
        return []
    
    print(df.head)
    # sec_code列とdoc_id列が存在するか確認
    if 'sec_code' not in df.columns or 'doc_id' not in df.columns:
        print(f"Error: Required columns (sec_code and/or doc_id) not found in the file")
        return []
    
    # sec_code列を文字列型に変換
    df['sec_code'] = df['sec_code'].astype(str)
    
    # 指定されたsec_codeに一致する行のdoc_idを抽出
    matching_doc_ids = df[df['sec_code'] == target_sec_code]['doc_id'].tolist()
    
    return matching_doc_ids

# 使用例
if __name__ == "__main__":
    file_path = "/root/src/RawData/data/interim/2017/documents.csv"  # メタデータファイルのパスを指定
    target_sec_code = "72030"  # 抽出したいsec_codeを指定
    
    result = extract_doc_ids(file_path, target_sec_code)
    if result:
        print(f"Extracted doc_ids for sec_code {target_sec_code}:")
        print(result)
    else:
        print(f"No matching doc_ids found for sec_code {target_sec_code}")