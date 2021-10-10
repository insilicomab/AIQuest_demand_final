# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_engineering():
    
    # データの読み込み
    pre_train = pd.read_csv('./processed_data/pre_train.csv')
    test = pd.read_csv('./data/test.csv')
    category_names = pd.read_csv('./data/category_names.csv')
    item_categories = pd.read_csv('./data/item_categories.csv')
    sales_history = pd.read_csv('./data/sales_history.csv')
    time_group = pd.read_csv('./processed_data/time_group.csv')
    
    '''
    テストデータの加工
    '''
    
    # カラム名の変更
    test = test.rename(columns={'商品ID': 'id',
                                '店舗ID':'store_id'
                                })
    
    # month_biningカラムの作成
    test['month_bining'] = 23
    
    '''
    学習データとテストデータの結合
    '''
    
    df = pd.concat([pre_train, test], sort=False).reset_index(drop=True)
    
    # 不要なカラムの削除
    df = df.drop(['index'], axis=1)
    
    '''
    返品データの修正
    '''
    
    # 返品データを修正する
    for i in range(len(df)):
        if df['sales'][i] < 0: # train['sales']のi行目の要素抽出
            df['sales'][i] = 0
    
    print(df['sales'].value_counts())
    print(df['sales'].isnull().sum())
    
    '''
    ラグ特徴量の作成
    '''
    
    # 12ヶ月前の売上データ
    lag12before = df.copy()
    lag12before['month_bining'] = lag12before['month_bining']+12
    lag12before = lag12before.rename(columns={'sales':'sales_before_12'})
    
    # 11ヶ月前の売上データ
    lag11before = df.copy()
    lag11before['month_bining'] = lag11before['month_bining']+11
    lag11before = lag11before.rename(columns={'sales':'sales_before_11'})
    
    # 10ヶ月前の売上データ
    lag10before = df.copy()
    lag10before['month_bining'] = lag10before['month_bining']+10
    lag10before = lag10before.rename(columns={'sales':'sales_before_10'})
    
    # 9ヶ月前の売上データ
    lag9before = df.copy()
    lag9before['month_bining'] = lag9before['month_bining']+9
    lag9before = lag9before.rename(columns={'sales':'sales_before_9'})
    
    # 8ヶ月前の売上データ
    lag8before = df.copy()
    lag8before['month_bining'] = lag8before['month_bining']+8
    lag8before = lag8before.rename(columns={'sales':'sales_before_8'})
    
    # 7ヶ月前の売上データ
    lag7before = df.copy()
    lag7before['month_bining'] = lag7before['month_bining']+7
    lag7before = lag7before.rename(columns={'sales':'sales_before_7'})
    
    # 6ヶ月前の売上データ
    lag6before = df.copy()
    lag6before['month_bining'] = lag6before['month_bining']+6
    lag6before = lag6before.rename(columns={'sales':'sales_before_6'})
    
    # 5ヶ月前の売上データ
    lag5before = df.copy()
    lag5before['month_bining'] = lag5before['month_bining']+5
    lag5before = lag5before.rename(columns={'sales':'sales_before_5'})
    
    # 4ヶ月前の売上データ
    lag4before = df.copy()
    lag4before['month_bining'] = lag4before['month_bining']+4
    lag4before = lag4before.rename(columns={'sales':'sales_before_4'})
    
    # 3ヶ月前の売上データ
    lag3before = df.copy()
    lag3before['month_bining'] = lag3before['month_bining']+3
    lag3before = lag3before.rename(columns={'sales':'sales_before_3'})
    
    # 2ヶ月前の売上データ
    lag2before = df.copy()
    lag2before['month_bining'] = lag2before['month_bining']+2
    lag2before = lag2before.rename(columns={'sales':'sales_before_2'})
    
    # ラグ特徴量の追加
    df = pd.merge(df, lag12before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag11before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag10before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag9before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag8before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag7before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag6before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag5before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag4before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag3before, on=['month_bining', 'id', 'store_id'], how='left')
    df = pd.merge(df, lag2before, on=['month_bining', 'id', 'store_id'], how='left')
    print(df.groupby('month_bining').agg({'sales_before_12': 'count'}))
    
    '''
    商品カテゴリ、年月の情報を付与
    '''
    
    # カラム名の変更
    category_names = category_names.rename(columns={'商品カテゴリID':'category_id',
                                                    '商品カテゴリ名':'category_name'})
    
    item_categories = item_categories.rename(columns={'商品ID': 'id',
                                                      '商品カテゴリID':'category_id'})
    
    sales_history = sales_history.rename(columns={'商品ID': 'id',
                                                  '日付':'datetime',
                                                  '店舗ID':'store_id',
                                                  '商品価格':'price',
                                                  '売上個数':'sales'
                                                  })
    
    # 商品カテゴリIDの付与
    df = pd.merge(df, item_categories, on='id', how='left')
    
    # 商品カテゴリ名の付与
    df = pd.merge(df, category_names, on='category_id', how='left')
    
    # カテゴリを分ける
    df['category'] = df['category_name'].apply(lambda x : x.split(' - ')[0])
    
    # 年、月の付与
    df = pd.merge(df, time_group, on='month_bining', how='left')
    
    '''
    売上個数のTarget Encoding
    '''
    
    # 商品IDのTarget Encoding
    sales_mean = sales_history.groupby('id').agg({'sales':np.mean}).reset_index()
    
    # カラム名の変更
    sales_mean = sales_mean.rename(columns={'sales':'sales_mean'})
    
    # Target Encodingを付与
    df = pd.merge(df, sales_mean, on='id', how='left')
    
    
    '''
    欠損値の補完
    '''
    
    # 欠損値の確認
    print(df.isnull().sum())
    
    df['sales_before_12'] = df['sales_before_12'].fillna(0)
    df['sales_before_11'] = df['sales_before_11'].fillna(0)
    df['sales_before_10'] = df['sales_before_10'].fillna(0)
    df['sales_before_9'] = df['sales_before_9'].fillna(0)
    df['sales_before_8'] = df['sales_before_8'].fillna(0)
    df['sales_before_7'] = df['sales_before_7'].fillna(0)
    df['sales_before_6'] = df['sales_before_6'].fillna(0)
    df['sales_before_5'] = df['sales_before_5'].fillna(0)
    df['sales_before_4'] = df['sales_before_4'].fillna(0)
    df['sales_before_3'] = df['sales_before_3'].fillna(0)
    df['sales_before_2'] = df['sales_before_2'].fillna(0)
    print(df.isnull().sum())
    
    '''
    不要なカラムの削除
    '''
    
    # カラムの削除
    df = df.drop(['category_name'], axis=1)
    print(df.columns)
    
    '''
    データフレームの保存
    '''
    
    df.to_csv('./processed_data/processed_train_test_df.csv', header=True, index=False)


if __name__ == '__main__':
    feature_engineering()