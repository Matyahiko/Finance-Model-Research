import pandas as pd
import mplfinance as mpf
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# データの読み込み
data = pd.read_csv("ProcessedData/japan-all-stock-prices_filtered_7203.csv")

# 日付をインデックスに設定
data['日付'] = pd.to_datetime(data['日付'], format='%Y%m%d')
data = data.set_index('日付')

# 目的変数と説明変数の分離
X = data.drop(["株価","時価総額（百万円）"],axis=1)
y = data["株価"]

# 数値データのみ選択
numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
X = X[numeric_cols]

# 時間的な分割
tscv = TimeSeriesSplit(n_splits=5)

# 予測値を保存するDataFrame
predictions_df = pd.DataFrame(index=data.index, columns=['Open', 'High', 'Low', 'Close', 'Actual'])
predictions_df[['Open', 'High', 'Low', 'Close']] = data[['始値', '高値', '安値', '株価']]
predictions_df['Actual'] = y

for i, (train_index, test_index) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # モデルの構築とフィッティング
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # テストデータで予測
    y_pred = model.predict(X_test)

    # 評価指標の計算
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Fold {i} - Mean Squared Error: {mse:.2f}")
    print(f"Fold {i} - R-squared: {r2:.2f}")

    # 特徴量の重要度を抽出
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print(f"\nFold {i} - 特徴量の重要度:")
    print(feature_importances.sort_values(ascending=False))

    # 予測値を保存
    predictions_df.loc[y_test.index, f'Predicted_{i}'] = y_pred

# mplfinanceを使用してプロットし、画像として保存する
apdict = mpf.make_addplot(predictions_df.drop(columns=['Actual', 'Open', 'High', 'Low', 'Close']), type='line')
mpf.plot(predictions_df, type='candle', style="charles", title="Actual vs. Predicted Stock Prices",
         ylabel="Stock Price", addplot=apdict,
         savefig=dict(fname="fig/actual_vs_predicted_all.png", dpi=300, format="png"))