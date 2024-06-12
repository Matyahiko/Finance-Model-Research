
from xml.dom.minidom import Document
from transformers import BertJapaneseTokenizer, BertForSequenceClassification,DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import os
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[0m'



def calculate_metrics(true_labels, predictions):
    cm = multilabel_confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    return cm, precision, recall, f1

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

def create_labeled_dataframe(df):
    labeled_df = pd.DataFrame(columns=["sentence", "label"])
    
    for _, row in df.iterrows():
        if pd.notna(row["朝食_ポジティブ"]) and row["朝食_ポジティブ"] == 1:
            label = 1
        elif pd.notna(row["朝食_ネガティブ"]) and row["朝食_ネガティブ"] == 1:
            label = 0
        else:
            label = 2
            
        new_row = pd.DataFrame({"sentence": [row["レビュー文"]], "label": [label]})
        labeled_df = pd.concat([labeled_df, new_row], ignore_index=True)
    
    return labeled_df

def tokenize_function(examples):
    #前処理
    #paddingとmaxlengthで挙動がおかしいのでDataCollatorを使用
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, padding=True, max_length=512)
    return {"input_ids": tokenized_inputs["input_ids"], "labels": examples["label"],"attention_mask": tokenized_inputs["attention_mask"]}

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
print(undersampled_df)

# データの分割
train_data, test_data = train_test_split(undersampled_df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

#print(dataset)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(YELLOW+f"device:{device}" + RESET)

num_gpus = torch.cuda.device_count()
print( YELLOW + f"Available GPUs: {num_gpus}" + RESET)

batch_size = 50

# BERTモデルとトークナイザーの読み込み
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  
# if num_gpus > 1:
#     model = nn.DataParallel(model)
# model.to(device)
model.to(device)


#datacollatorの設定
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#train validation testのサブセットを渡す
# データセットの前処理
encoded_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])

print(encoded_dataset["train"].column_names)

# 必要な列のみを選択
#上と重複してる部分があるけどまあいいや
train_dataset = encoded_dataset["train"].select_columns(["input_ids", "labels"])
validation_dataset = encoded_dataset["validation"].select_columns(["input_ids", "labels"])
test_dataset = encoded_dataset["test"].select_columns(["input_ids", "labels"])

# データローダーの作成
train_dataloader = DataLoader(encoded_dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=data_collator)
validation_dataloader = DataLoader(encoded_dataset["validation"], batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=data_collator)
test_dataloader = DataLoader(encoded_dataset["test"], batch_size=batch_size, shuffle=True,drop_last=True,collate_fn=data_collator)


#DebugInfo
#torchの形状を変更したので、一部機能しない
#トークンの復元チェックは機能していないけど、想定通りの動作なので放置
if False :
    sample_num = 1
    print(GREEN + f"encoded_dataset sentence sample{sample_num}: \n" + str(encoded_dataset["train"]["input_ids"][sample_num]) +"\n" + RESET)
    print(GREEN + f"decoded_dataset sentence sample{sample_num}: \n" + tokenizer.decode(encoded_dataset["train"]["input_ids"][sample_num])+"\n" + RESET)

    print(GREEN + f"encoded_dataset sentence sample{sample_num+1}: \n" + str(encoded_dataset["train"]["input_ids"][sample_num+1]) +"\n" + RESET)
    print(GREEN + f"decoded_dataset sentence sample{sample_num+1}: \n" + tokenizer.decode(encoded_dataset["train"]["input_ids"][sample_num+1])+"\n" + RESET)

    print(GREEN + f"batch size \n {train_dataloader.batch_size} \n"+RESET)
    train_true = encoded_dataset["train"]["labels"].count(0)
    train_false = encoded_dataset["train"]["labels"].count(1)
    
    test_true = encoded_dataset["test"]["labels"].count(0)
    test_false = encoded_dataset["test"]["labels"].count(1)
    
    train_total = train_true + train_false
    train_true_ratio = train_true / train_total
    train_false_ratio = train_false / train_total

    test_total = test_true + test_false
    test_true_ratio = test_true / test_total
    test_false_ratio = test_false / test_total
    
    print(GREEN + f"train_true: {train_true} train_false: {train_false} train_total: {train_total} \n"+RESET)
    print(GREEN + f"train_true_ratio: {train_true_ratio} train_false_ratio: {train_false_ratio} \n"+RESET)
    print(GREEN + f"test_true: {test_true} test_false: {test_false} test_total: {test_total} \n"+RESET)
    print(GREEN + f"test_true_ratio: {test_true_ratio} test_false_ratio: {test_false_ratio} \n"+RESET)
    
   
    for batch in train_dataloader:
        print(GREEN + f"batch : \n{batch} \n"+RESET)
        print(GREEN + f"batch column: \n {batch.keys()} \n"+RESET)
        input_ids = batch["input_ids"]
        print(GREEN + f"input_ids: \n{type(input_ids)} \n"+RESET)
        decoded_text = [tokenizer.decode(id[0]) for id in input_ids]
        print(type(input_ids[1][1]))
        bun = []
        doc = [bun.append(strings)  for strings in decoded_text]
        print(GREEN+f"After Dataloader Decoded : {bun}"+RESET)
        labels = batch["labels"]
        print("Labels:", labels,"\nLabels shape" ,labels.shape)
        break
    
    i = 0
    for k,batch in enumerate(train_dataloader):
        i+=k
        
    print(i)       
        

# オプティマイザーの設定
optimizer = AdamW(model.parameters(), lr=2e-5)

#損失関数の設定
criterion = torch.nn.CrossEntropyLoss()

# ファインチューニングのループ
num_epochs = 1
for epoch in range(num_epochs):
    # 訓練
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
       # print(GREEN + f"outputs{outputs}"+RESET)
        loss = outputs.loss
       # print(GREEN + f"loss{loss}"+RESET)
        loss.backward()
        optimizer.step()
        
    # 検証
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            validation_loss += outputs.loss.item()
    
    validation_loss /= len(validation_dataloader)
    print(f"Epoch {epoch+1} - Validation Loss: {validation_loss:.6f}")

# テストデータでの評価
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.extend(outputs.logits.argmax(dim=-1).tolist())
        true_labels.extend(labels.tolist())

print("Test predictions:", predictions)

cm, precision, recall, f1 = calculate_metrics(true_labels, predictions)

# modelの保存
if isinstance(model, nn.DataParallel):
    model = model.module
os.makedirs("research/AspectBertModel", exist_ok=True)
model.save_pretrained("research/AspectBertModel")