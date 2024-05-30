import numpy as np
import webbrowser
import os

def tensor_to_html(tensor):
    html = "<html><body><table>"
    html += "<tr><th>Index</th><th>Value</th></tr>"
    for i, value in enumerate(tensor):
        html += f"<tr><td>{i}</td><td>{value}</td></tr>"
    html += "</table></body></html>"
    return html

def main():
    tensor = np.load("log/random_news_tensor.npy", allow_pickle=True)
    html = tensor_to_html(tensor)

    os.makedirs("log", exist_ok=True)  # ディレクトリが存在しない場合は作成
    with open("log/tensor.html", "w") as f:
        f.write(html)
    webbrowser.open("file://" + os.path.abspath("log/tensor.html"))

if __name__ == "__main__":
    main()