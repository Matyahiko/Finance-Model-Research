import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import torch
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import h5py
import json
from joblib import Memory

from CryptoCatboostTest import test

GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[0m'


cache_dir = './cache/crypto'
memory = Memory(cache_dir, verbose=0)

#@memory.cache
class ParallelFeaturesTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, num_features):
        # Separate target from features
        self.features = torch.FloatTensor((data.reset_index()).drop(['close_time','target'], axis=1).values)
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
    print(train_df)

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
def read_csv_with_dtypes(filepath):
    # まず型指定なしで読み込む
    df = pd.read_csv(filepath, index_col=0)
    # 数値列のみをfloat32に変換
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].astype('float32')
    return df


def objective(trial):
    
        # パラメータの定義
    batch_size = trial.suggest_int('batch_size', 16, 128)
    sequence_length = trial.suggest_int('sequence_length', 3, 48)
    

    # Optunaによるパラメータ探索
    
   # ハイパーパラメータの定義
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 100.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "random_strength": trial.suggest_uniform("random_strength", 1e-9, 10),
        #"bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 100.0),
        "loss_function": "RMSE" , # 明示的に損失関数を指定
        "task_type" : "GPU"
    }
    
    
    
    train_loader, val_loader,test_loader = data_loader(
    train_df, 
    val_df, 
    test_df,
    sequence_length,
    num_features,
    batch_size
    )

    # 全データの取得
    train_data, train_labels = get_full_data(train_loader)
    val_data, val_labels = get_full_data(val_loader)
    # print(type(train_data))
    # print(train_data)
    # print(train_data.shape)
    
    # モデルの学習
    model = CatBoostRegressor(**params)
    model.fit(train_data, 
            train_labels, 
            eval_set=(val_data, val_labels), 
            early_stopping_rounds=15, 
            verbose=10)

    # 検証データでの予測
    preds = model.predict(val_data)

    # 評価指標の計算（ここではRMSEを使用）
    rmse = np.sqrt(mean_squared_error(val_labels, preds))

    return rmse  # 最小化問題を最大化問題に変換


filename = "BTC-JPY_5min_2021-2024"
# CSVから読み込む
train_df = read_csv_with_dtypes(f"crypto/processed/{filename}_train.csv")
val_df = read_csv_with_dtypes(f"crypto/processed/{filename}_val.csv")
test_df = read_csv_with_dtypes(f"crypto/processed/{filename}_test.csv")

num_features = len(train_df.columns)-1


#debug
if False:
    print(GREEN + f"num_features : {num_features}" + RESET)

    # データローダーからイテレータを取得
    loader_iter = iter(train_loader)

    print(GREEN + f"DataLoader Info:" + RESET)
    print(GREEN + f"Batch size: {test_loader.batch_size}" + RESET)
    print(GREEN + f"Number of batches: {len(test_loader)}" + RESET)

    first_batch = next(loader_iter)
    features, labels = first_batch

    print(GREEN + f"\nFirst batch shape: {features.shape}" + RESET)
    print(GREEN + f"Data type: {features.dtype}" + RESET)

    # サンプルデータの表示
    print(GREEN + "\nSample data (first 5 elements of first item in batch):" + RESET)
    print(GREEN + f"{features[0, :5]}" + RESET)

    print(GREEN + "label" + RESET)
    print(GREEN + f"{labels}" + RESET)
    
    train_data, train_labels = get_full_data(train_loader)
    val_data, val_labels = get_full_data(val_loader)
    
    print(GREEN + f"Train data shape: {train_data.shape}" + RESET)
    print(GREEN + f"Train labels shape: {train_labels.shape}" + RESET)
    print(GREEN + f"Val data shape: {val_data.shape}" + RESET)
    print(GREEN + f"Val labels shape: {val_labels.shape}" + RESET)
    print(GREEN + f"Train data type: {train_data.dtype}" + RESET)
    print(GREEN + f"Train labels type: {train_labels.dtype}" + RESET)
    print(GREEN + f"Train data range: {train_data.min()} - {train_data.max()}" + RESET)
    print(GREEN + f"Train labels range: {train_labels.min()} - {train_labels.max()}" + RESET)
    print(GREEN + f"Infinite values in train data: {np.isinf(train_data).any()}" + RESET)
    print(GREEN + f"NaN values in train data: {np.isnan(train_data).any()}" + RESET)
    print(GREEN + f"Infinite values in train labels: {np.isinf(train_labels).any()}" + RESET)
    print(GREEN + f"NaN values in train labels: {np.isnan(train_labels).any()}" + RESET)
    

# Optunaによる最適化の実行
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# 最良のトライアルの表示
print("Best trial:")
trial = study.best_trial
print("  Value: ", -trial.value)  # RMSEの値（元の最小化問題の値）
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_sequence_length = trial.params["sequence_length"]
best_batch_size = trial.params["batch_size"]

train_loader, val_loader,test_loader = data_loader(
train_df, 
val_df, 
test_df,
best_sequence_length,
num_features,
best_batch_size
)

# CatBoostRegressor用のパラメータを準備
catboost_params = trial.params.copy()
catboost_params.pop("sequence_length", None)
catboost_params.pop("batch_size", None)


# 最適化されたパラメータでモデルを再学習
best_model = CatBoostRegressor(**catboost_params)
train_data, train_labels = get_full_data(train_loader)
val_data,val_labels = get_full_data(val_loader)
history = best_model.fit(train_data, 
        train_labels, 
        eval_set=(val_data, val_labels), 
        early_stopping_rounds=15, 
        verbose=10)


# 学習曲線の描画
plt.figure(figsize=(10, 6))
plt.plot(history.evals_result_['learn']['RMSE'], label='Train')
plt.plot(history.evals_result_['validation']['RMSE'], label='Validation')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('CatBoost Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig("crypto/fig/CatBoost_Learning_Curve.png")

# モデルの保存
best_model.save_model("crypto/models/best_catboost_model.cbm")

# 設定をJSONファイルとして保存
config = {
    "sequence_length": best_sequence_length,
    "batch_size": best_batch_size
}

with open("crypto/models/config.json", "w") as f:
    json.dump(config, f)

# 特徴量の寄与率を計算
feature_importance = best_model.get_feature_importance()
feature_names = best_model.feature_names_

# 特徴量の寄与率を降順にソート
sorted_idx = np.argsort(feature_importance)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = feature_importance[sorted_idx]

# 上位20個の特徴量の寄与率をプロット
plt.figure(figsize=(10, 8))
plt.barh(range(20), sorted_importance[-20:])
plt.yticks(range(20), sorted_features[-20:])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('crypto/fig/feature_importance.png')
plt.close()

# 特徴量の寄与率を表示
print("\nFeature Importance:")
for feature, importance in zip(sorted_features[-20:], sorted_importance[-20:]):
    print(f"{feature}: {importance:.4f}")
    
test()