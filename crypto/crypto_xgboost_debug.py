import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# CSVファイルを読み込む
df = pd.read_csv('/root/src/src/crypto/procesed/btc_technical_analysis_train.csv', index_col='close_time', parse_dates=True)
df['macd_cross'] = df['macd_cross'].map({-1: 0, 0: 1, 1: 2})

# 特徴量と目的変数を定義
features = [col for col in df.columns if col != 'macd_cross']
target = 'macd_cross'

# 時系列交差検証の設定
"""
全データ数: 311805
学習データ数: 249444
テストデータ数: 62361
"""
n_splits = 24900
tscv = TimeSeriesSplit(n_splits=n_splits)

# XGBoostモデルの設定
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',  # GPUを使用する
    'gpu_id': "0,1",  # 使用するGPUのID
}

# パラメータグリッドの設定
param_grid = {
    'lambda': [0.1, 0.5, 1.0],
    'alpha': [0.1, 0.5, 1.0],
    'min_child_weight': [1, 5, 10],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
}

# モード選択
mode = input("実行モードを選択してください（train/test）: ")

if mode == 'train':
    # グリッドサーチの実行
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(**params),
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(df[features], df[target])

    # 最適なパラメータと最高の性能を表示
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    # 最適なモデルを取得
    best_model = grid_search.best_estimator_

    # 交差検証の実行
    accuracies = []
    for train_index, val_index in tscv.split(df):
        X_train, X_val = df[features].iloc[train_index], df[features].iloc[val_index]
        y_train, y_val = df[target].iloc[train_index], df[target].iloc[val_index]
        
        # データセットの準備
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # モデルの学習
        best_model.fit(dtrain)
        
        # モデルの予測
        y_pred_val = best_model.predict(dval)
        y_pred_val = np.argmax(y_pred_val, axis=1)
        
        accuracy = accuracy_score(y_val, y_pred_val)
        accuracies.append(accuracy)

    # モデルを保存
    best_model.save_model('src/crypto/models/xgboost_model.json')

    # 特徴量の重要度を表示
    feature_importances = best_model.get_booster().get_score(importance_type='gain')
    feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    print("Feature Importances:")
    for feature, importance in feature_importances:
        print(f"{feature}: {importance}")

    print(f"\nMean Validation Accuracy: {np.mean(accuracies):.4f}")

elif mode == 'test':
    # 保存されたモデルを読み込む
    if os.path.exists('src/crypto/models/xgboost_model.json'):
        model = xgb.Booster()
        model.load_model('src/crypto/models/xgboost_model.json')

        # テストデータを読み込む
        test_data = pd.read_csv('/root/src/src/crypto/procesed/btc_technical_analysis_test.csv', index_col='close_time', parse_dates=True)
        X_test = test_data[features]
        y_test = test_data[target]

        # テストデータでの予測精度を評価
        dtest = xgb.DMatrix(X_test)
        y_pred_test = model.predict(dtest)
        y_pred_test = np.argmax(y_pred_test, axis=1)

        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))
    else:
        print("モデルが見つかりません。先に学習モードで実行してください。")
else:
    print("無効なモードが指定されました。'train'または'test'を指定してください。")