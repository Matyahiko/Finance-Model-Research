
from xml.dom.minidom import Document
from transformers import BertJapaneseTokenizer, BertForSequenceClassification,DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import os
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[0m'

def calculate_metrics(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return cm, precision, recall, f1

# # データセットの読み込み
# dataset = load_dataset("llm-book/wrime-sentiment")
# #データセットの保存
# os.makedirs("research/SentimentData", exist_ok=True)
# dataset.save_to_disk("research/SentimentData")

dataset = load_from_disk("research/SentimentData")
#print(dataset)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(YELLOW+f"device:{device}" + RESET)


num_gpus = torch.cuda.device_count()
print( YELLOW + f"Available GPUs: {num_gpus}" + RESET)

batch_size = 25

# BERTモデルとトークナイザーの読み込み
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  
# if num_gpus > 1:
#     model = nn.DataParallel(model)
# model.to(device)
model.to(device)

#前処理
#paddingとmaxlengthで挙動がおかしいのでDataCollatorを使用
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, padding=True, max_length=512)
    return {"input_ids": tokenized_inputs["input_ids"], "labels": examples["label"],"attention_mask": tokenized_inputs["attention_mask"]}

#datacollatorの設定
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#train validation testのサブセットを渡す
# データセットの前処理
encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["sentence","user_id","datetime","label"])

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
if True :
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

# ファインチューニングのループ
num_epochs = 3
for epoch in range(num_epochs):
    # 訓練
    model.train()
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 検証
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        validation_loss += loss.item()
    
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
        outputs = model(input_ids=input_ids, labels=labels)
        predictions.extend(outputs.logits.argmax(dim=-1).tolist())
        true_labels.extend(labels.tolist())

print("Test predictions:", predictions)

cm, precision, recall, f1 = calculate_metrics(true_labels, predictions)

# modelの保存
# DataParallelを使っている場合は、model.moduleでラップされているのでラップを解いて保存
if isinstance(model, nn.DataParallel):
    model = model.module
os.makedirs("research/SentimentBertModel", exist_ok=True)
model.save_pretrained("research/SentimentBertModel")