import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


from xml.dom.minidom import Document
from transformers import BertJapaneseTokenizer, BertForSequenceClassification,DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import os
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class TextAutoencoder(nn.Module):
    def __init__(self, encoder_model_name, hidden_size, vocab_size):
        super(TextAutoencoder, self).__init__()
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Create new decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get encoder output
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Pass through decoder
        decoder_output = self.decoder(encoder_output)
        
        return decoder_output

# Generate text
def generate_text(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    
    for _ in range(max_length):
        outputs = model(input_ids, attention_mask)
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1,1))], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0])



#sample
df = pd.read_csv("research_SecuritiesReport/combined_docs_72030.tsv",sep="\t")

print(df["doc_id"])
grouped_df = df.groupby(["doc_id"])
print(grouped_df)

# Example usage
encoder_model_name = "bert-base-uncased"
hidden_size = 768  # BERT's hidden size
vocab_size = 30522  # BERT's vocabulary size

model = TextAutoencoder(encoder_model_name, hidden_size, vocab_size)
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.decoder.parameters())  # Only optimize decoder
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, vocab_size), input_ids.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


