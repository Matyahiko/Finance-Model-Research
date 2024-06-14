from transformers import BertJapaneseTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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
    label_groups = df.groupby(label_column)
    min_samples = min(label_groups.size())
    undersampled_groups = []
    for _, group in label_groups:
        undersampled_group = resample(group, replace=False, n_samples=min_samples, random_state=42)
        undersampled_groups.append(undersampled_group)
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
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, padding=True, max_length=512)
    return {"input_ids": tokenized_inputs["input_ids"], "labels": examples["label"], "attention_mask": tokenized_inputs["attention_mask"]}

df = pd.read_csv("research/RakutenData/travel_aspect_sentiment/travel_aspect_sentiment.tsv", sep="\t", header=0)
labeled_df = create_labeled_dataframe(df)
undersampled_df = undersample_majority_class(labeled_df, "label")

_, test_data = train_test_split(undersampled_df, test_size=0.2, random_state=42)

test_dataset = Dataset.from_pandas(test_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 訓練済みモデルとトークナイザーの読み込み
MODEL_PATH = "research/AspectBertModel"
TOKENIZER_PATH = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = BertJapaneseTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# テストデータの前処理
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

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