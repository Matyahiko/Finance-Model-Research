import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def rolling_window_split(data, window_size, step_size):
    for i in range(0, len(data) - window_size, step_size):
        yield data.index[i:i + window_size], data.index[i + window_size:i + window_size + step_size]
00
# CSVファイルを読み込む
df = pd.read_csv("/root/src/src/crypto/procesed/btc_15min_technical_analysis_train.csv", index_col="close_time", parse_dates=True)
df["macd_cross"] = df["macd_cross"].map({-1: 0, 0: 1, 1: 2})

# 特徴量と目的変数を定義
features = ["open_price", "high_price", "low_price", "close_price", "SMA5", "SMA10", "SMA20", "SMA50", "SMA100", "SMA200", "upper_band", "middle_band", "lower_band", "macd", "macdsignal", "macdhist", "RSI", "slowk", "slowd", "ADX", "CCI", "ATR", "ROC", "Williams %R"]
target = "macd_cross"

# XGBoostモデルの設定
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    #"eval_metric": "mlogloss",
}
#Best parameters:  {"alpha": 0.5, "colsample_bytree": 0.8, "lambda": 0.1, "max_depth": 5, "min_child_weight": 1, "subsample": 1.0}
# パラメータグリッドの設定
# param_grid = {
#     "lambda": [0.1, 0.5, 1.0],
#     "alpha": [0.1, 0.5, 1.0],
#     "min_child_weight": [1, 5, 10],
#     "max_depth": [3, 5, 7],
#     "subsample": [0.5, 0.8, 1.0],
#     "colsample_bytree": [0.5, 0.8, 1.0],
# }
param_grid = {
    "lambda": [0.1],
    "alpha": [0.5],
    "min_child_weight": [1],
    "max_depth": [5],
    "subsample": [1.0],
    "colsample_bytree": [0.8],
}

# ローリングウィンドウの設定
window_size = 12  # 適切な値に調整してください
step_size = 1  # 適切な値に調整してください

# モード選択
mode = input("実行モードを選択してください（train/test）: ")

if mode == "train":
    start_time = time.time()  # 学習の開始時間を記録
    k = 0
    # ローリングウィンドウでデータを分割
    accuracies = []
    for train_index, val_index in rolling_window_split(df, window_size, step_size):
        X_train, X_val = df[features].loc[train_index], df[features].loc[val_index]
        y_train, y_val = df[target].loc[train_index], df[target].loc[val_index]
        k+=1
        #debug
        print("=" * 50)
        print("Train data:")
        print("Index range:", train_index[0], "to", train_index[-1])
        print("Shape:", X_train.shape)
        print("shape:",y_train.shape)
        print("=" * 50)
        print("Validation data:")
        print("Index range:", val_index[0], "to", val_index[-1])
        print("Shape:", X_val.shape)
        print("shape:",y_val.shape)
        print("=" * 50)
        
        # if not y_train.shape == y_val.shape and not k < 13:
        #     break
        
        # # グリッドサーチの実行
        # grid_search = GridSearchCV(
        #     estimator=xgb.XGBClassifier(**params),
        #     param_grid=param_grid,
        #     cv=3,  # 適切な値に調整してください
        #     scoring="accuracy",
        #     verbose=2,
        #     n_jobs=-1
        # )
        # grid_search.fit(X_train, y_train,
        #                 eval_set=[(X_val, y_val)],
        #                 eval_metric="merror",
        #                 early_stopping_rounds=10,  # アーリーストッピングのラウンド数を指定
        #                 verbose=True)
        
        # # 最適なモデルを取得
        # best_model = grid_search.best_estimator_
        
        # # 検証データでの予測精度を評価
        # y_pred_val = best_model.predict(X_val)
        # accuracy = accuracy_score(y_val, y_pred_val)
        # accuracies.append(accuracy)

        # end_time = time.time()  # 学習の終了時間を記録
        # training_time = end_time - start_time  # 学習にかかった時間を計算

        # # 最適なパラメータと最高の性能を表示
        # print("Best parameters: ", grid_search.best_params_)
        # print("Best score: ", grid_search.best_score_)

        # # モデルを保存
        # best_model.save_model("src/crypto/models/xgboost_model.json")

        # # 特徴量の重要度を表示
        # feature_importances = best_model.get_booster().get_score(importance_type="gain")
        # feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        # print("Feature Importances:")
        # for feature, importance in feature_importances:
        #     print(f"{feature}: {importance}")
        
        # print(f"\nMean Validation Accuracy: {np.mean(accuracies):.4f}")
        # print(f"Training Time: {training_time:.2f} seconds")  # 学習にかかった時間を表示

elif mode == "test":
    # 保存されたモデルを読み込む
    if os.path.exists("src/crypto/models/xgboost_model.json"):
        model = xgb.Booster()
        model.load_model("src/crypto/models/xgboost_model.json")

        # テストデータを読み込む
        test_data = pd.read_csv("/root/src/src/crypto/procesed/btc_15min_technical_analysis_test.csv", index_col="close_time", parse_dates=True)
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
    print("無効なモードが指定されました。\"train\"または\"test\"を指定してください。")
