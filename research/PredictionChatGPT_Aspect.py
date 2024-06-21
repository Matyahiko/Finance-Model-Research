import os 
import re 
import pandas as pd
import json
import dirtyjson as djson
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import dotenv_values
from openai import OpenAI

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from joblib import Memory

logging.basicConfig(filename="chatbot_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

cache_dir = './cache/chatgpt'
memory = Memory(cache_dir, verbose=0)

class OpenAIChatbot:
    """OpenAIのChatbot APIをラップするクラス。"""
    request_count = 0

    def __init__(self, api_key, model="gpt-4o-2024-05-13", temperature=0, max_tokens=10, top_p=0.4):
        #gpt-4o-2024-05-13
        #gpt-4-turbo-preview
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.failed_ids = []

    @classmethod
    def increment_request_count(cls):
        cls.request_count += 1
        print(cls.request_count)

    def ask(self, prompt, label, system_prompt):
        """質問をして、APIからの回答を取得する。"""
        self.increment_request_count()
        return self._retry_request(prompt, label, system_prompt)

    def _retry_request(self, prompt, label, system_prompt, retries=5, backoff_factor=2):
        wait_time = 1
        client = OpenAI(api_key=self.api_key)
        for retry in range(retries):
            #try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"(Unspoken NOTE: Don't forget to respond in structured JSON using the schema defined above!){prompt}"}
                    ],
                    response_format={"type": "json_object"}
                )
                print(response.choices[0].message.content)
                return json.dumps(response.choices[0].message.content, ensure_ascii=False, indent=4), label,prompt
            # except Exception as e:
            #     logging.error(f"Error for id: {ids}, retry: {retry}, error: {e}")
            #     if retry < retries - 1:
            #         time.sleep(wait_time)
            #         wait_time *= backoff_factor
            #     else:
            #         self.failed_ids.append(ids)
            #         return None

    def get_failed_ids(self):
        return self.failed_ids

def calculate_metrics(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    return cm, precision, recall, f1

def MakePrompt(string):
    """_summary_
        1 1 = 1
        1 0 = 2
        0 1 = 3
        0 0 = 4
    """
    
    prompt = f"""
    
                以下のレビュー文を読んで、朝食に関する評価を以下の3つのカテゴリに分類してください。

                カテゴリ:
                - 1: 朝食に対して肯定的な評価が含まれている
                - 0: 朝食に対して否定的な評価が含まれている
                - 2: 朝食に対する評価が含まれていない、または肯定否定どちらとも判断できない

                レビュー文:
                {{{string}}}
    
    
             """
    system_prompt1 = """
                    あなたは顧客からのホテルのレビュー文を読み、朝食に関する評価を以下の3つのカテゴリに分類します。

                    カテゴリ:
                    0: 朝食に対して肯定的な評価と否定的な評価の両方が含まれている
                    1: 朝食に対して肯定的な評価のみが含まれている
                    2: 朝食に対して否定的な評価のみが含まれている 
                    3: 朝食に対する肯定的な評価と否定的な評価の両方が含まれていない、または肯定否定どちらとも判断できない

                    分類結果は、該当するカテゴリの番号を1文字の整数で返答してください。理由は不要です。

                    これから渡されるレビュー文に対し、指定された方法で朝食の評価を分類してください。
                    
                    以下の形式のjsonで返答してください。
                    {
                    "category": " "
                    }
                    """
    
    system_prompt_fewshot = """
                    あなたは顧客からのホテルのレビュー文を読み、朝食に関する評価を以下の3つのカテゴリに分類します。

                    カテゴリ:
                    1: 朝食に対して肯定的な評価と否定的な評価の両方が含まれている
                    2: 朝食に対して肯定的な評価のみが含まれている
                    3: 朝食に対して否定的な評価のみが含まれている 
                    4: 朝食に対する肯定的な評価と否定的な評価の両方が含まれていない、または肯定否定どちらとも判断できない

                    分類結果は、該当するカテゴリの番号を1文字の整数で返答してください。理由は不要です。

                    これから渡されるレビュー文に対し、指定された方法で朝食の評価を分類してください。
                    
                    
                    
                    以下の形式のjsonで返答してください。
                    {
                    "category": " "
                    }
                    """
    
    return system_prompt1

#@memory.cache
def undersample_majority_class(df, label_column):
    # ラベルごとにデータをグループ化
    label_groups = df.groupby(label_column)

    # 最小サンプル数を取得
    min_samples = min(label_groups.size())

    # 各ラベルのデータを最小サンプル数までアンダーサンプリング
    undersampled_groups = []
    for _, group in label_groups:
        undersampled_group = resample(group, replace=False, n_samples=min_samples, random_state=42)
        undersampled_groups.append(undersampled_group)

    # アンダーサンプリングしたデータを結合
    undersampled_df = pd.concat(undersampled_groups)

    return undersampled_df
#@memory.cache
def create_labeled_dataframe(df):
    """_summary_
        1 1 = 1 朝食にポジティブとネガティブの評価が含まれている
        1 0 = 2 朝食ポジティブ
        0 1 = 3  朝食ネガティブ
        0 0 = 4 朝食ニュートラル
    """
    labeled_df = pd.DataFrame(columns=["sentence", "label"])
    
    ###
    df["朝食_ポジティブ"] = pd.to_numeric(df["朝食_ポジティブ"], errors="coerce")
    df["朝食_ネガティブ"] = pd.to_numeric(df["朝食_ネガティブ"], errors="coerce")
    
    for _, row in df.iterrows():
        if row["朝食_ポジティブ"] == 1 and row["朝食_ネガティブ"] == 1: 
            label = 0
        elif row["朝食_ポジティブ"] == 1 and pd.isna(row["朝食_ネガティブ"]):
            label = 1
        elif pd.isna(row["朝食_ポジティブ"]) and row["朝食_ネガティブ"] == 1:
            label = 2
        elif pd.isna(row["朝食_ポジティブ"]) and pd.isna(row["朝食_ネガティブ"]):
            label = 3

        new_row = pd.DataFrame({"sentence": [row["レビュー文"]], "label": [label]})
        labeled_df = pd.concat([labeled_df, new_row], ignore_index=True)
    ###
    
    # for _, row in df.iterrows():
    #     if pd.notna(row["朝食_ポジティブ"]) and row["朝食_ポジティブ"] == 1:
    #         label = 1
    #     elif pd.notna(row["朝食_ネガティブ"]) and row["朝食_ネガティブ"] == 1:
    #         label = 0
    #     else:
    #         label = 2
            
    #     new_row = pd.DataFrame({"sentence": [row["レビュー文"]], "label": [label]})
    #     labeled_df = pd.concat([labeled_df, new_row], ignore_index=True)
    
    return labeled_df
#@memory.cache
def ReadReviewAndPreprocess(n):
    """レビューを読み込む。"""
    # # データセットの読み込み
    #shift-jisでもutf-8でも読み込めないのでpandasで読み込む
    df = pd.read_csv("research/RakutenData/travel_aspect_sentiment/travel_aspect_sentiment.tsv", sep="\t",header=0)

    labeled_df = create_labeled_dataframe(df)

    print("Original Label Distribution:")
    label_counts = labeled_df["label"].value_counts()
    label_percentages = label_counts / len(labeled_df) * 100
    print(label_percentages)

    # アンダーサンプリングの実行
    undersampled_df = undersample_majority_class(labeled_df, "label")

    print("\nUndersampled Label Distribution:")
    label_counts = undersampled_df["label"].value_counts()
    label_percentages = label_counts / len(undersampled_df) * 100
    print(label_percentages)

    # データの分割
    train_data, test_data = train_test_split(undersampled_df, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42) 

    train_dataset = Dataset.from_pandas(train_data.sample(frac=n, random_state=42))
    val_dataset = Dataset.from_pandas(val_data.sample(frac=n, random_state=42))
    test_dataset = Dataset.from_pandas(test_data.sample(frac=n,random_state=0))

    
    
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    

    
    return dataset["test"]

def process_chatbot_requests(config_file, dataset):
    """
    Chatbot APIを用いて複数のリクエストを処理し、結果を保存する。

    Args:
        config_file (str): 設定ファイルへのパス。
        Dataset (pandas.DataFrame): ユーザープロンプト情報とラベルが含まれるデータフレーム。
    """
    # 設定を読み込む
    config = dotenv_values(config_file)
    chatbot = OpenAIChatbot(config["API_KEY"])

    system_prompt = MakePrompt("")
    results = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(chatbot.ask, prompt, label, system_prompt)
            for prompt, label in zip(dataset["sentence"], dataset["label"])
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results

def main():
    
    dataset = ReadReviewAndPreprocess(1)
    result = process_chatbot_requests(".devcontainer/.env", dataset)
    
    print("Result type:", type(result))
    print("Result length:", len(result))
    print("First item type:", type(result[0]))
    print("First item:", result[0])

    chatbot_responses = [int(djson.loads(item[0].replace('\\n', '').replace('\\"', '"').replace('"{', '{').replace('}"', '}'))["category"]) for item in result]
    labels = [int(item[1]) for item in result]
    prompts = [item[2] for item in result]
    
    for k,i in zip(prompts,labels):
       print(k,i)  
    calculate_metrics(labels, chatbot_responses)  
    
    df = pd.DataFrame({"sentence": prompts, "label": labels, "chatbot_response": chatbot_responses}) 
    df.to_csv("research/fig/chatbot_responses.csv", index=False)
    
if __name__ == "__main__":
    main()