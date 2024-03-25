import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def generate_embeddings(text, model):

    embeddings = model.encode(text)
    return embeddings.tolist()

def plot_embeddings(df):
    embeddings = np.array(df["embeddings"].tolist())
    title = df["Title"].tolist()
    id = df["ID"].tolist()
    
        # TitleとIDを組み合わせてツールチップ用のテキストリストを作成
    tooltips = ["ID: " + str(id_) + "<br>Title: " + title for id_, title in zip(id, title)]

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings)

    # 散布図
    fig = go.Figure(data=go.Scatter(x=tsne_results[:, 0], 
                                    y=tsne_results[:, 1],
                                    mode='markers',
                                    text=tooltips,  # ツールチップとして表示するテキスト
                                    hoverinfo="text"))  # ホバー時にtextの内容のみを表示
    
    fig.update_layout(title='visualization of embeddings',
                      xaxis=dict(title='t-SNE1'),
                      yaxis=dict(title='t-SNE2'))
    
    # グラフを表示
    fig.show()
    
    # 画像保存 (この部分はPlotlyのオフライン環境では動作しないことがあるため注意)
    fig.write_html("fig/tsne.html")
    
    


def main():
    folder_path = "raw_data/news"
    tsv_files = get_tsv_files_in_directory(folder_path)
    combined_df = pd.concat([pd.read_csv(file, sep="\t", header=0) for file in tsv_files], axis=0)
    ###前処理###
    #ユニークIDを付与
    combined_df["ID"] = range(1, len(combined_df) + 1) 
    #重複を削除
    combined_df.drop_duplicates(subset=["Title"], inplace=True)
    
    
    

    model = SentenceTransformer("models/intfloat_multilingual-e5-large/")
    combined_df['embeddings'] = combined_df['Text'].apply(lambda text: generate_embeddings(text, model))
    # print(combined_df['embeddings'].head())
    # print(combined_df['embeddings'].shape)
    #埋め込みサイズの確認
    print(combined_df['embeddings'].iloc[0])
    print(len(combined_df['embeddings'].iloc[0]))
    
    plot_embeddings(combined_df)
    

if __name__ == "__main__":
    main()
