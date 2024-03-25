from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import os

# 保存先のディレクトリを指定
save_directory = "/root/src/models/"

# 保存先のディレクトリが存在しない場合は作成
os.makedirs(save_directory, exist_ok=True)

# モデルをダウンロードして保存
#model = SentenceTransformer("intfloat/multilingual-e5-large", cache_folder=save_directory)

# トークナイザーをダウンロードして保存
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct", cache_dir=save_directory)
tokenizer.save_pretrained(save_directory)

# モデルをダウンロードして保存
model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct", cache_dir=save_directory)
model.save_pretrained(save_directory)