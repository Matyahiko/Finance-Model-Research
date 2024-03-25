import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_tsv_files_in_directory(directory):
    tsv_files = []
    items = os.listdir(directory)
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            tsv_files.extend(get_tsv_files_in_directory(item_path))
        elif item.endswith(".tsv"):
            tsv_files.append(item_path)
    return tsv_files

def generate_embeddings(text, model, tokenizer):
    max_length = 4096
    batch_dict = tokenizer(text, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    batch_dict["input_ids"] = [[input_id] + [tokenizer.eos_token_id] for input_id in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()

def plot_embeddings(df):
    embeddings = np.array(df["embeddings"].tolist())
    title = df["Title"].tolist()
    id = df["ID"].tolist()
    tooltips = ["ID: " + str(id_) + "<br>Title: " + title for id_, title in zip(id, title)]
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings)
    fig = go.Figure(data=go.Scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], mode="markers", text=tooltips, hoverinfo="text")) 
    fig.update_layout(title="visualization of embeddings", xaxis=dict(title="t-SNE1"), yaxis=dict(title="t-SNE2"))
    fig.show()
    fig.write_html("fig/tsne.html")

def main():
    folder_path = "raw_data/news" 
    tsv_files = get_tsv_files_in_directory(folder_path)
    combined_df = pd.concat([pd.read_csv(file, sep="\t", header=0) for file in tsv_files], axis=0)
    combined_df["ID"] = range(1, len(combined_df) + 1)
    combined_df.drop_duplicates(subset=["Title"], inplace=True)
    tokenizer = AutoTokenizer.from_pretrained("models/e5-mistral-7b-instruct")
    model = AutoModel.from_pretrained("models/e5-mistral-7b-instruct")
    combined_df["embeddings"] = combined_df["Text"].apply(lambda text: generate_embeddings(text, model, tokenizer))
    print(len(combined_df["embeddings"].iloc[0]))
    plot_embeddings(combined_df)

if __name__ == "__main__":
    main()