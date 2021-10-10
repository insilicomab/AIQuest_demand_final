# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def rf():
    
    '''
    初期設定
    '''
    
    DATA_PATH = './processed_data/processed_train_test_df.csv'
    SAMPLESUB_PATH = './data/sample_submission.csv'
    SUB_PATH = './submit/demand_RandomForest.csv'
    
    # データの読み込み
    df = pd.read_csv(DATA_PATH)
    print(df.isnull().sum())
    print(df.dtypes)
    
    # object型の変数の取得
    categories = df.columns[df.dtypes == 'object']
    print(categories)
    
    # label Encoding
    for cat in categories:
        le = LabelEncoder() 
        print(cat)
        
        df[cat].fillna('missing', inplace=True)
        le = le.fit(df[cat])
        df[cat] = le.transform(df[cat])
        # LabelEncoderは数値に変換するだけであるため、最後にastype('category')としておく
        df[cat] = df[cat].astype('category') 
    
    # trainとtestに分割
    train = df[~df['sales'].isnull()]
    test = df[df['sales'].isnull()]
    
    # 説明変数と目的変数を指定
    X_train = train.drop(['sales'], axis=1)
    Y_train = train['sales']
    X_test = test.drop(['sales'], axis=1)
    
    '''
    モデルの構築と評価
    '''
    
    # ライブラリのインポート
    from sklearn.ensemble import RandomForestRegressor as rf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.2, random_state=0,
                                                          shuffle=False)
    
    model = rf(n_estimators=50,
               random_state=1234)   
        
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f'RMSE:{rmse}')
    
    #特徴量の重要度
    feature = model.feature_importances_
    label = X_train.columns[0:] #特徴量の名前
    indices = np.argsort(feature)[::1] #特徴量の重要度順(降順)
    
    # プロット
    x = range(len(feature))
    y = feature[indices]
    y_label = label[indices]
    plt.barh(x, y, align = 'center')
    plt.yticks(x, y_label)
    plt.xlabel("importance_num")
    plt.ylabel("label")
    plt.show()
    
    """
    予測精度：
    RMSE:1.9025693884048573
    """
    
    '''
    テストデータの予測
    '''
    
    # テストデータにおける予測
    pred = model.predict(X_test)
    
    '''
    提出
    '''
    
    # 提出用サンプルの読み込み
    sub = pd.read_csv(SAMPLESUB_PATH, header=None)
    
    # カラム1の値を置き換え
    sub[1] = pred
    
    # CSVファイルの出力
    sub.to_csv(SUB_PATH, header=None, index=False)
    
    """
    スコア：
    2.9828661
    """


if __name__ == '__main__':
    rf()