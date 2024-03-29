import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import webbrowser

def plot_embeddings(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings)

    fig = go.Figure(data=go.Scatter(x=tsne_results[:, 0], y=tsne_results[:, 1],
                                    mode='markers'))
    fig.update_layout(title='Visualization of Embeddings',
                      xaxis=dict(title='t-SNE1'),
                      yaxis=dict(title='t-SNE2'))

    fig.write_html("embeddings_plot.html")
    webbrowser.open("embeddings_plot.html")

def main():
    embeddings = np.load("log/random_news_tensor.npy", allow_pickle=True)
    plot_embeddings(embeddings)

if __name__ == "__main__":
    main()