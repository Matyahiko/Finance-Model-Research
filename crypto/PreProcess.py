import json
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from CryptoCatboostTrain import cat_train

from joblib import Memory

cache_dir = './cache/crypto'
memory = Memory(cache_dir, verbose=0)

def standardize_dataframe(df, exclude_columns=['close_time']):
    """
    DataFrame内の数値列を標準化する関数。
    指定された列（デフォルトは'close_time'）は標準化から除外される。

    Parameters:
    df (pd.DataFrame): 標準化するDataFrame
    exclude_columns (list): 標準化から除外する列名のリスト

    Returns:
    pd.DataFrame: 標準化されたDataFrame
    dict: 各列のScalerオブジェクト
    """
    # 入力DataFrameのコピーを作成
    df_scaled = df.copy()
    
    # 数値列を特定（exclude_columnsに含まれない列のみ）
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
    
    # 各列に対してStandardScalerを適用
    scalers = {}
    for col in columns_to_scale:
        scaler = StandardScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    
    return df_scaled, scalers

def analyze_target_distribution(df: pd.DataFrame, name:str, target_column: str = 'target', save_dir: str = 'crypto/fig' ):
    """
    Analyze and visualize the distribution of target labels in a DataFrame.
    
    :param df: pandas DataFrame containing the dataset
    :param target_column: name of the target column (default is 'target')
    """
    # 基本的な統計情報
    print("Target Distribution Summary:")
    print(df[target_column].describe())
    
    # 値の頻度
    value_counts = df[target_column].value_counts()
    print("\nValue Counts:")
    print(value_counts)
    
    # 分布の可視化
    plt.figure(figsize=(10, 6))
    
    # ヒストグラム
    plt.subplot(2, 1, 1)
    sns.histplot(df[target_column], kde=True)
    plt.title('Histogram of Target Values')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    
    # 箱ひげ図
    plt.subplot(2, 1, 2)
    sns.boxplot(x=df[target_column])
    plt.title('Boxplot of Target Values')
    plt.xlabel('Target Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_target_distribution.png'))
    plt.close()
    
    # クラスバランス（カテゴリカルデータの場合）
    if df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[target_column])
        plt.title('Class Balance')
        plt.xlabel('Target Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_dir, f'{name}_class_balance.png'))
        plt.close()

def split_time_series_data(df, test_size=0.2, val_size=0.2):
    """
    時系列データを学習、検証、テストデータに分割する関数
    
    :param df: 分割するデータフレーム
    :param test_size: テストデータの割合
    :param val_size: 検証データの割合（学習データ内での割合）
    :return: 学習データ、検証データ、テストデータ
    """
    # テストデータを分割（最新のデータ）
    train_val_size = 1 - test_size
    train_val_len = int(len(df) * train_val_size)
    
    train_val_df = df[:train_val_len]
    test_df = df[train_val_len:]
    
    # 学習データを学習と検証に分割
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=False)
    
    # 結果の出力
    print(f"全データ数: {len(df)}")
    print(f"学習データ数: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"検証データ数: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"テストデータ数: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # データの連続性確認
    print("\nデータの連続性確認:")
    print(f"学習データの最後の日付: {train_df.index[-1]}")
    print(f"検証データの最初の日付: {val_df.index[0]}")
    print(f"検証データの最後の日付: {val_df.index[-1]}")
    print(f"テストデータの最初の日付: {test_df.index[0]}")
    
    return train_df, val_df, test_df

#@memory.cache
def load_data(file_path, split_point=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    filename = file_path.split("/")[-1].split(".")[0]
    
    df = pd.DataFrame(data)
    df['close_time'] = pd.to_datetime(df['close_time'], format='%Y/%m/%d %H:%M:%S')
    
    if split_point:
        df = df[:split_point]
    
    print(f"データ数: {len(df)}")
    
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("close_time", inplace=True)
    
    return df,filename

#@memory.cache
def add_technical_indicators(df):
    open_price, high_price, low_price, close_price = df["open_price"], df["high_price"], df["low_price"], df["close_price"]
    
    df["SMA5"], df["SMA10"], df["SMA20"] = talib.SMA(close_price, 5), talib.SMA(close_price, 10), talib.SMA(close_price, 20)
    df["SMA50"], df["SMA100"], df["SMA200"] = talib.SMA(close_price, 50), talib.SMA(close_price, 100), talib.SMA(close_price, 200)
    df["upper_band"], df["middle_band"], df["lower_band"] = talib.BBANDS(close_price, timeperiod=20)
    df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
    df["RSI"] = talib.RSI(close_price, timeperiod=14)
    df["slowk"], df["slowd"] = talib.STOCH(high_price, low_price, close_price, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df["ADX"] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
    df["CCI"] = talib.CCI(high_price, low_price, close_price, timeperiod=14)
    df["ATR"] = talib.ATR(high_price, low_price, close_price, timeperiod=14)
    df["ROC"] = talib.ROC(close_price, timeperiod=10)
    df["Williams %R"] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
    
    return df

def remove_multicollinearity(train_data, test_data, threshold=0.7):
    """
    多重共線性を示す特徴量を削除する関数

    Args:
        train_data (pandas.DataFrame): 学習データのデータフレーム
        test_data (pandas.DataFrame): テストデータのデータフレーム
        threshold (float, optional): 多重共線性の閾値. Defaults to 0.7.

    Returns:
        tuple: 多重共線性が認められる特徴量を削除後の学習データとテストデータのタプル
    """
    corr_matrix = train_data.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    columns_to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > threshold)]

    train_data_cleaned = train_data.drop(columns=columns_to_drop)
    test_data_cleaned = test_data.drop(columns=columns_to_drop)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('crypto/fig/correlation_heatmap.png')
    plt.close()

    return train_data_cleaned, test_data_cleaned

def calculate_target_variable(df, holding_period=1, profit_threshold=0.02):
    """
    エントリールールを探索するため、各時間ステップで指定期間ホールドした際の利益率を計算し、
    利益率が閾値を超える場合を１、超えない場合には０にする関数。

    Args:
        df (pandas.DataFrame): OHLC データを含むデータフレーム
        holding_period (int): ホールド期間（デフォルト: 24、5分足データの場合2時間）
        profit_threshold (float): 利益率の閾値（デフォルト: 0.005）

    Returns:
        pandas.DataFrame: エントリールールとその利益率を追加したデータフレーム
    """
    
    # 将来の価格を計算
    future_close = df["close_price"].shift(-holding_period)
    future_high = df["high_price"].rolling(window=holding_period, min_periods=1).max().shift(-holding_period)
    future_low = df["low_price"].rolling(window=holding_period, min_periods=1).min().shift(-holding_period)

    # 利益率を計算
    long_profit_ratio = (future_high - df["open_price"]) / df["open_price"]
    short_profit_ratio = (df["open_price"] - future_low) / df["open_price"]

    df["target"] = long_profit_ratio
    
    # ターゲット変数を生成
    # df['target'] = np.where(
    #     (long_profit_ratio > profit_threshold) | (short_profit_ratio > profit_threshold),
    #     1,
    #     0
    # )

    # 利益率も追加
    #df['long_profit_ratio'] = long_profit_ratio
    #df['short_profit_ratio'] = short_profit_ratio

    # NaNを0に置換（データ終端の処理）
    #df[['target', 'long_profit_ratio', 'short_profit_ratio']] = df[['target', 'long_profit_ratio', 'short_profit_ratio']].fillna(0)

    return df

def main():
    # データの読み込み
    df,filename = load_data('RawData/btc/BTC-JPY_15min_2021-2024.json', split_point=100000)
    
    # 特徴量の追加
    df = add_technical_indicators(df)
    
    # ターゲット変数の計算
    df = calculate_target_variable(df)
    
    # 不要な列を削除
    #df.drop("close_time", axis=1, inplace=True)
    df.dropna(inplace=True)
    
    df,scalers = standardize_dataframe(df)
    
    
    # 学習データとテストデータに分割
    train_df, val_df, test_df = split_time_series_data(df)
    
    analyze_target_distribution(train_df,"train")
    analyze_target_distribution(val_df,"val")
    analyze_target_distribution(test_df,"test")
    
    # 多重共線性が認められる特徴量を削除
    #train_df_cleaned, test_df_cleaned = remove_multicollinearity(train_df, test_df)
    
    print(len(df.columns))
    print(df.columns)
    
    print(f"多重共線性が認められる特徴量を削除後の学習データ数: {train_df.shape}")
    print(f"多重共線性が認められる特徴量を削除後のテストデータ数: {test_df.shape}")
    print(f"削除された特徴量: {set(train_df.columns) - set(train_df.columns)}")
    

        
    for df in [train_df, val_df, test_df]:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_columns] = df[numeric_columns].astype('float32')

    # インデックスを保持しつつCSVファイルに保存
    train_df.to_csv(f"crypto/processed/{filename}_train.csv", index=True, float_format='%.8f')
    val_df.to_csv(f"crypto/processed/{filename}_val.csv", index=True, float_format='%.8f')
    test_df.to_csv(f"crypto/processed/{filename}_test.csv", index=True, float_format='%.8f')
    
    cat_train(filename)

if __name__=="__main__":
    main()