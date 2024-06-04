
from xml.dom.minidom import Document
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import os
from sklearn.metrics import confusion_matrix

# # データセットの読み込み
# dataset = load_dataset("llm-book/wrime-sentiment")
# #データセットの保存
# os.makedirs("research/SentimentData", exist_ok=True)
# dataset.save_to_disk("research/SentimentData")

dataset = load_from_disk("research/SentimentData")
#print(dataset)

# BERTモデルとトークナイザーの読み込み
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # num_labelsはラベルの数に置き換えてください

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("device:", device)

#前処理
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=50)

#train validation testのサブセットを渡す
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 必要な列のみを選択
train_dataset = encoded_dataset["train"].select_columns(["input_ids", "label"])
validation_dataset = encoded_dataset["validation"].select_columns(["input_ids", "label"])
test_dataset = encoded_dataset["test"].select_columns(["input_ids", "label"])

# データローダーの作成
train_dataloader = DataLoader(encoded_dataset["train"], batch_size=50, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(encoded_dataset["validation"], batch_size=50, shuffle=True, drop_last=True)
test_dataloader = DataLoader(encoded_dataset["test"], batch_size=50, shuffle=True,drop_last=True)


#DebugInfo
if False :
    sample_num = 1
    GREEN = '\033[32m'
    RESET = '\033[0m'
    print(GREEN + f"encoded_dataset sentence samaple{sample_num}: \n" + str(encoded_dataset["train"]["input_ids"][sample_num]) +"\n" + RESET)
    print(GREEN + f"decoded_dataset sentence sample{sample_num}: \n" + tokenizer.decode(encoded_dataset["train"]["input_ids"][sample_num])+"\n" + RESET)

    print(GREEN + f"encoded_dataset sentence samaple{sample_num+1}: \n" + str(encoded_dataset["train"]["input_ids"][sample_num+1]) +"\n" + RESET)
    print(GREEN + f"decoded_dataset sentence sample{sample_num+1}: \n" + tokenizer.decode(encoded_dataset["train"]["input_ids"][sample_num+1])+"\n" + RESET)

    print(GREEN + f"batch size \n {train_dataloader.batch_size} \n"+RESET)
   
    for batch in train_dataloader:
        #print(GREEN + f"batch : \n{batch} \n"+RESET)
        
        print(GREEN + f"batch column: \n {batch.keys()} \n"+RESET)
        input_ids = batch["input_ids"]
        print(GREEN + f"input_ids: \n{type(input_ids)} \n"+RESET)
        decoded_text = [tokenizer.decode(id[0]) for id in input_ids]
        print(type(input_ids[1][1]))
        bun = []
        doc = [bun.append(strings)  for strings in decoded_text]
        print(GREEN+f"After Dataloader Decoded : {bun}"+RESET)
        labels = batch["label"]
        print("Labels:", labels,"\nLabels shape" ,labels.shape)
        break


# オプティマイザーの設定
optimizer = AdamW(model.parameters(), lr=2e-5)

# ファインチューニングのループ
num_epochs = 10
for epoch in range(num_epochs):
    # 訓練
    model.train()
    for batch in train_dataloader:
        input_ids = torch.stack(batch["input_ids"]).to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 検証
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        input_ids = torch.stack(batch["input_ids"]).to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        validation_loss += outputs.loss.item()

    validation_loss /= len(validation_dataloader)
    print(f"Epoch {epoch+1} - Validation Loss: {validation_loss:.6f}")

# テストデータでの評価
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = torch.stack(batch["input_ids"]).to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        predictions.extend(outputs.logits.argmax(dim=-1).tolist())
        true_labels.extend(labels.tolist())

print("Test predictions:", predictions)

# 混同行列の計算
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)