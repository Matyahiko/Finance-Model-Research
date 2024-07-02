import os
import re
import pandas as pd

def clean_text(text):
    # 改行を空白に置き換える
    text = re.sub(r'\n', '', text)
    # 連続する空白を単一の空白に置き換える
    text = re.sub(r'\s+', '', text)
    # 先頭と末尾の空白を削除する
    text = text.strip()
    return text

def load_files_to_dataframe(directory, file_prefix):
    data = []
    
    for filename in os.listdir(directory):
        if filename.startswith(file_prefix) and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            # テキストをクリーニングする
            cleaned_text = clean_text(text)
            
            file_suffix = filename.replace(file_prefix, "").replace(".txt", "")
            data.append([file_suffix, cleaned_text])
    
    df = pd.DataFrame(data, columns=["file_suffix", "text"])
    return df

# 使用例
directory = "RawData/data/interim/2017/docs"
file_prefix = "S100DA2Y_"

df = load_files_to_dataframe(directory, file_prefix)
print(df)
df.to_csv(f"research_SecuritiesReport/{file_prefix}.tsv", sep="\t")