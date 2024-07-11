import os
import re
import pandas as pd

def extract_doc_ids(file_path, target_sec_code):
    try:
        df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty")
        return []
    
    if 'sec_code' not in df.columns or 'doc_id' not in df.columns:
        print(f"Error: Required columns (sec_code and/or doc_id) not found in the file")
        return []
    
    df['sec_code'] = df['sec_code'].astype(str)
    matching_doc_ids = df[df['sec_code'] == target_sec_code]['doc_id'].tolist()
    return matching_doc_ids

def clean_text(text):
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()

def load_files_to_dataframe(directory,doc_id):
    data = []
    file_prefix = f"{doc_id[0]}_"
    
    for filename in os.listdir(directory):
        if filename.startswith(file_prefix) and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            # テキストをクリーニングする
            cleaned_text = clean_text(text)
            
            file_suffix = filename.replace(file_prefix, "").replace(".txt", "")
            data.append([file_suffix, cleaned_text])

    return pd.DataFrame(data, columns=["doc_id", "text"])

def process_documents(years, target_sec_code):
    all_data = []
    #metadata_file = f"/root/src/RawData/data/interim/2014/documents.csv"
    
    #doc_id=str(doc_ids[0])
    for year in years:
        metadata_file = f"/root/src/RawData/data/interim/{year}/documents.csv"
        #複数ある場合を考えてristになってる
        doc_ids = extract_doc_ids(metadata_file, target_sec_code)
        print(doc_ids)
        docs_directory = f"/root/src/RawData/data/interim/{year}/docs"
        
        df = load_files_to_dataframe(docs_directory,doc_ids)
        df['year'] = year
        all_data.append(df)
       
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_file = f"research_SecuritiesReport/combined_docs_{target_sec_code}.tsv"
        combined_df.to_csv(output_file, sep="\t", index=False)
        print(f"Combined data saved to {output_file}")
        return combined_df
    else:
        print("No data found for the specified years and sec_code")
        return None

# 使用例
if __name__ == "__main__":
    years = range(2014, 2019)  # 2014年から2018年まで
    target_sec_code = "72030"  # 抽出したいsec_codeを指定
    
    result = process_documents(years, target_sec_code)
    if result is not None:
        print(result)
    
    result.to_csv(f"research_SecuritiesReport/AnalysisbyIndustry/{target_sec_code}.csv",sep="\t")