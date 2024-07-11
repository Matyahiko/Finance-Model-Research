import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import h5py
from joblib import Memory

import matplotlib.pyplot as plt
import japanize_matplotlib 

GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[0m'


cache_dir = './cache/crypto'
memory = Memory(cache_dir, verbose=0)

#@memory.cache
class ParallelFeaturesTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, num_features):
        # Separate target from features
        self.features = torch.FloatTensor(data.drop('target', axis=1).values)
        self.targets = torch.FloatTensor(data['target'].values)
        self.sequence_length = sequence_length
        self.num_features = num_features - 1  # Subtract 1 for the target column

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, index):
        feature_sequence = self.features[index:index + self.sequence_length]
        # Reshape the feature sequence
        feature_sequence = feature_sequence.transpose(0, 1).reshape(-1)
        
        # Get the corresponding target (use the last value in the sequence)
        # Reshape the sequence to [f1_1, f2_1, ..., f1_2, f2_2, ..., f1_n, f2_n, ...]   
        target = self.targets[index + self.sequence_length - 1]

        return feature_sequence, target

#@memory.cache
def data_loader(train_df, val_df, test_df, sequence_length, num_features, batch_size):
    train_dataset = ParallelFeaturesTimeSeriesDataset(train_df, sequence_length, num_features)
    val_dataset = ParallelFeaturesTimeSeriesDataset(val_df, sequence_length, num_features)
    test_dataset = ParallelFeaturesTimeSeriesDataset(test_df, sequence_length, num_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_full_data(data_loader):
    full_data = []
    full_labels = []
    for batch in data_loader:
        features, labels = batch
        full_data.append(features)
        full_labels.append(labels)
    return torch.cat(full_data, dim=0).numpy(), torch.cat(full_labels, dim=0).numpy()

#@memory.cache
def load_data_from_hdf5(filename):
    
   with h5py.File(f"crypto/processed/{filename}.h5", 'r') as hf:
        # 文字列のリストとして読み込む
        columns = list(hf.attrs['columns'])
        
        train_data = {col: hf['train'][col][:] for col in columns}
        val_data = {col: hf['val'][col][:] for col in columns}
        test_data = {col: hf['test'][col][:] for col in columns}
        
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)
    
   return train_df, val_df, test_df



filename = "BTC-JPY_15min_2021-2024"
train_df, val_df, test_df = load_data_from_hdf5(filename)

sequence_length = 12
num_features = len(train_df.columns)-1
batch_size = 20

train_loader, val_loader, test_loader = data_loader(
    train_df, 
    val_df, 
    test_df,
    sequence_length,
    num_features,
    batch_size
)

#debug
if True:
    print(f"num_features : {num_features}")
        # データローダーからイテレータを取得
    loader_iter = iter(train_loader)
    print(f"DataLoader Info:")
    print(f"Batch size: {test_loader.batch_size}")
    print(f"Number of batches: {len(test_loader)}")
    first_batch = next(loader_iter)
    features, labels = first_batch
    print(f"\nFirst batch shape: {features.shape}")
    print(f"Data type: {features.dtype}")
    # サンプルデータの表示
    print("\nSample data (first 5 elements of first item in batch):")
    print(features[0, :5])
    print("label")
    print(labels)

best_model = CatBoostRegressor()
best_model.load_model("crypto/models/best_catboost_model.cbm")

# テストデータの準備
test_data, test_labels = get_full_data(test_loader)

# テストデータでの予測
test_predictions = best_model.predict(test_data)

# 評価指標の計算
rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
mae = mean_absolute_error(test_labels, test_predictions)
r2 = r2_score(test_labels, test_predictions)

# 結果の表示
print("テストデータでの性能評価:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R2 Score: {r2:.6f}")

# 実際の値と予測値の散布図
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(test_labels, test_predictions, alpha=0.5)
plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('テストデータでの実際の値 vs 予測値')
plt.tight_layout()
plt.savefig('crypto/fig/test_predictions_scatter.png')
plt.close()

# 残差プロット
residuals = test_labels - test_predictions
plt.figure(figsize=(10, 6))
plt.scatter(test_predictions, residuals, alpha=0.5)
plt.xlabel('予測値')
plt.ylabel('残差')
plt.title('残差プロット')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('crypto/fig/residuals_plot.png')
plt.close()

print("散布図と残差プロットが crypto/fig/ ディレクトリに保存されました。")