import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import h5py
import json
from joblib import Memory

import matplotlib.pyplot as plt
import japanize_matplotlib 

from sklearn.linear_model import LinearRegression

from PreProcess import analyze_target_distribution

GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[0m'


cache_dir = './cache/crypto'
memory = Memory(cache_dir, verbose=0)

def calculate_metrics(test_labels, test_predictions):
    """評価指標を計算する関数"""
    rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
    mae = mean_absolute_error(test_labels, test_predictions)
    r2 = r2_score(test_labels, test_predictions)
    return rmse, mae, r2

def display_results(rmse, mae, r2):
    """結果を表示する関数"""
    print("テストデータでの性能評価:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2 Score: {r2:.6f}")

def create_scatter_plot(test_labels, test_predictions, save_path):
    """散布図を作成し保存する関数"""
    plt.figure(figsize=(10, 6))
    plt.scatter(test_labels, test_predictions, alpha=0.5)
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
    plt.xlabel('実際の値')
    plt.ylabel('予測値')
    plt.title('テストデータでの実際の値 vs 予測値')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_residual_plot(test_labels, test_predictions, save_path):
    """残差プロットを作成し保存する関数"""
    residuals = test_labels - test_predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(test_predictions, residuals, alpha=0.5)
    plt.xlabel('予測値')
    plt.ylabel('残差')
    plt.title('残差プロット')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_test_vs_predictions(test_labels, test_predictions, fig_size=(10, 8), save_path=None):
    """
    テストデータの実際のラベルと予測値の散布図を作成する関数

    Parameters:
    test_labels (array-like): 実際のテストラベル
    test_predictions (array-like): モデルによる予測値
    fig_size (tuple): 図のサイズ。デフォルトは(10, 8)
    save_path (str, optional): 図を保存するパス。指定しない場合は表示のみ

    Returns:
    None
    """
    plt.figure(figsize=fig_size)
    
    # 散布図の作成
    plt.scatter(test_labels, test_predictions, c='blue', alpha=0.5, label='予測値')
    plt.scatter(test_labels, test_labels, c='red', alpha=0.5, label='実際の値')

    # 完全一致の線を追加
    max_value = max(np.max(test_labels), np.max(test_predictions))
    min_value = min(np.min(test_labels), np.min(test_predictions))
    plt.plot([min_value, max_value], [min_value, max_value], 'g--', label='完全一致線')

    # グラフの設定
    plt.xlabel('実際の値')
    plt.ylabel('予測値')
    plt.title('テストデータ：実際の値 vs 予測値')
    plt.legend()

    # グリッドの追加
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 保存パスが指定されている場合は保存、そうでなければ表示
    if save_path:
        plt.savefig(save_path)
        print(f"図が {save_path} に保存されました。")
    else:
        plt.show()

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

def read_csv_with_dtypes(filepath):
    # まず型指定なしで読み込む
    df = pd.read_csv(filepath, index_col=0)
    # 数値列のみをfloat32に変換
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].astype('float32')
    return df

def test():

    best_model = CatBoostRegressor()
    best_model.load_model("crypto/models/best_catboost_model.cbm")

    with open("crypto/models/config.json", "r") as f:
        loaded_config = json.load(f)

    filename = "BTC-JPY_5min_2021-2024"
    
    # CSVから読み込む
    train_df = read_csv_with_dtypes(f"crypto/processed/{filename}_train.csv")
    val_df = read_csv_with_dtypes(f"crypto/processed/{filename}_val.csv")
    test_df = read_csv_with_dtypes(f"crypto/processed/{filename}_test.csv")
    
    analyze_target_distribution(train_df,"train_test")
    analyze_target_distribution(val_df,"val_val")
    analyze_target_distribution(test_df,"test_test")

    sequence_length = loaded_config['sequence_length']
    num_features = len(train_df.columns)-1
    batch_size = loaded_config['batch_size']

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

    # テストデータの準備
    test_data, test_labels = get_full_data(test_loader)

    # テストデータでの予測
    test_predictions = best_model.predict(test_data)
    
    
    plot_test_vs_predictions(test_labels, test_predictions, fig_size=(12, 10), save_path='crypto/fig/test_predictions_scatter2.png')
    
    rmse, mae, r2 = calculate_metrics(test_labels, test_predictions)
    display_results(rmse, mae, r2)
    create_scatter_plot(test_labels, test_predictions, 'crypto/fig/test_predictions_scatter.png')
    create_residual_plot(test_labels, test_predictions, 'crypto/fig/residuals_plot.png')
    print("散布図と残差プロットが crypto/fig/ ディレクトリに保存されました。")
    print(test_labels)
    print(test_predictions)

if __name__ == "__main__":
    test()