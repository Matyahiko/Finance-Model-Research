import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import optuna



def rolling_window_split(data, window_size, step_size):
    """
    データをローリングウィンドウで分割するジェネレータ関数
    """
    for i in range(0, len(data) - window_size, step_size):
        yield data.index[i:i + window_size], data.index[i + window_size:i + window_size + step_size]

def load_and_preprocess_data(file_path):
    """
    CSVファイルを読み込み、前処理を行う関数
    """
    df = pd.read_csv(file_path, index_col="close_time", parse_dates=True)
    df["macd_cross"] = df["macd_cross"].map({-1: 0, 0: 1, 1: 2})
    
    # 目的変数を定義
    target = "label"

    # 目的変数以外の列を特徴量として選択
    features = [col for col in df.columns if col != target]
    
    return df, features, target

def train_model_timeseries(df, features, target, params, window_size, step_size):
   """
   モデルの学習を行う関数
   """
   start_time = time.time()  # 学習の開始時間を記録

   rmses = []
   model = CatBoostRegressor(**params)

   for train_index, val_index in rolling_window_split(df, window_size, step_size):
       X_train, X_val = df[features].loc[train_index], df[features].loc[val_index]
       y_train, y_val = df[target].loc[train_index], df[target].loc[val_index]

       model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

       y_pred_val = model.predict(X_val)
       rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
       rmses.append(rmse)

   end_time = time.time()  # 学習の終了時間を記録
   training_time = end_time - start_time  # 学習にかかった時間を計算

   mean_val_rmse = np.mean(rmses)
   print(f"\nMean Validation RMSE: {mean_val_rmse:.4f}")
   print(f"Training Time: {training_time:.2f} seconds")

   # モデルを保存
   model.save_model('crypto/models/model.py', format="python", export_parameters=None)

   return -mean_val_rmse  # Optunaは最小化を目的とするため、RMSEの負の値を返す

def test_model(test_data, features, target, window_size, step_size):
    """
    モデルのテストを行う関数
    """
    if os.path.exists("crypto/models/catboost_model.json"):
        model = CatBoostRegressor()
        model.load_model("crypto/models/catboost_model.json")

        X_test = test_data[features]
        y_test = test_data[target]

        test_rmses = []
        y_pred_test_list = []
        for test_index, _ in rolling_window_split(test_data, window_size, step_size):
            X_test_window = X_test.loc[test_index]
            y_test_window = y_test.loc[test_index[-1]]

            y_pred_test = model.predict(X_test_window.iloc[[-1]])
            y_pred_test_list.append(y_pred_test[0])

            rmse_test = np.sqrt(mean_squared_error([y_test_window], y_pred_test))
            test_rmses.append(rmse_test)

        print(f"\nMean Test RMSE: {np.mean(test_rmses):.4f}")

        # テストデータと予測データをプロット
        plot_test_prediction(y_test[window_size:], y_pred_test_list)

        # 混同行列の計算と出力
        y_test_binary = [1 if y > 0 else 0 for y in y_test[window_size:]]
        y_pred_test_binary = [1 if y > 0 else 0 for y in y_pred_test_list]
        cm = confusion_matrix(y_test_binary, y_pred_test_binary)
        print("\nConfusion Matrix:")
        print(cm)

        # プレシジョン、リコール、F1スコアの計算と出力
        precision = precision_score(y_test_binary, y_pred_test_binary)
        recall = recall_score(y_test_binary, y_pred_test_binary)
        f1 = f1_score(y_test_binary, y_pred_test_binary)
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    else:
        print("モデルが見つかりません。先に学習モードで実行してください。")

def plot_test_prediction(y_test, y_pred_test):
    """
    テストデータと予測データを同じ図にプロットする関数
    """
    y_pred_test_cal = []
    for i in y_pred_test:
        y_pred_test_cal.append(i*1)
        
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred_test_cal, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.title("Actual vs Predicted Return")

    # 最新の連番を付けて画像を保存
    #os.makedirs("/root/src/src/crypto/fig", exist_ok=True)
    existing_files = [f for f in os.listdir("/root/src/src/crypto/fig") if f.startswith("test_prediction")]
    if existing_files:
        latest_num = max([int(f.split(".")[0].split("_")[-1]) for f in existing_files])
        new_num = latest_num + 1
    else:
        new_num = 1
    
    plt.savefig(f"crypto/fig/test_prediction_{new_num}.png")
    plt.close()

def plot_learning_curve(rmses):
    """
    学習曲線を描画して保存する関数
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rmses) + 1), rmses, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Validation RMSE')
    plt.title('Learning Curve')
    plt.tight_layout()
    
    # 画像を保存
    os.makedirs("crypto/fig/learning_curve/", exist_ok=True)
    plt.savefig("crypto/fig/learning_curve/learning_curve.png")
    plt.close()

def objective(trial):
    

    # Optunaによるパラメータ探索
    params = {
        "objective": "RMSE",
        "eval_metric": "AUC",
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "iterations": trial.suggest_int("iterations", 100, 500),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 30),
        "random_seed": 42,
        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain", "Ordered"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }
    
    
    # ローリングウィンドウの設定
    window_size = 12
    step_size = 1
        
    # 学習データの読み込みと前処理
    df, features, target = load_and_preprocess_data("crypto/procesed/btc_1min_technical_analysis_train.csv")

    # モデルの学習
    best_score = train_model_timeseries(df, features, target, params, window_size, step_size)

    return best_score

if __name__ == "__main__":
   # モード選択
   mode = input("実行モードを選択してください（train/test）: ")

   if mode == "train":
       # Optunaによるパラメータ探索
       study = optuna.create_study(direction="maximize")
       study.optimize(objective, n_trials=100)

       # Optunaの可視化
       fig = optuna.visualization.plot_optimization_history(study)
       fig.show()

       fig = optuna.visualization.plot_parallel_coordinate(study)
       fig.show()

       fig = optuna.visualization.plot_contour(study)
       fig.show()

       fig = optuna.visualization.plot_param_importances(study)
       fig.show()

       print("Best trial:")
       trial = study.best_trial

       print("  Value: ", trial.value)
       print("  Params: ")
       for key, value in trial.params.items():
           print(f"    {key}: {value}")

       # 最適なパラメータでモデルを再学習
       best_params = trial.params
       window_size = 12
       step_size = 1
       df, features, target = load_and_preprocess_data("crypto/procesed/btc_1min_technical_analysis_train.csv")
       best_model = train_model_timeseries(df, features, target, best_params, window_size, step_size)

   elif mode == "test":
       window_size = 12
       step_size = 1
       # テストデータの読み込みと前処理
       test_data, features, target = load_and_preprocess_data("crypto/procesed/btc_1min_technical_analysis_test.csv")

       
       loaded_model = CatBoostRegressor()

       # モデルのテスト
       test_model(test_data, features, target, window_size, step_size, loaded_model)

       # 特徴量の重要度を可視化
       feature_importances = loaded_model.get_feature_importance(prettified=True)
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.barh(range(len(feature_importances)), feature_importances, align='center')
       ax.set_yticks(range(len(feature_importances)))
       ax.set_yticklabels(features)
       ax.set_xlabel('Feature Importance')
       ax.set_title('Feature Importances')
       plt.tight_layout()
       plt.show()