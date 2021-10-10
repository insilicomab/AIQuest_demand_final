# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_data_processing():
    
    # データの読み込み
    category_names = pd.read_csv('./data/category_names.csv')
    item_categories = pd.read_csv('./data/item_categories.csv')
    sales_history = pd.read_csv('./data/sales_history.csv')
    
    '''
    結合データの作成
    '''

    # データの結合
    productID_category = pd.merge(category_names, item_categories, on='商品カテゴリID', how='left')

    # 学習用データの作成
    merged_df = pd.merge(productID_category, sales_history, on='商品ID', how='left')

    # カラム名の変更
    merged_df = merged_df.rename(columns={'商品カテゴリID':'category_id',
                                          '商品カテゴリ名':'category_name',
                                          '商品ID': 'id',
                                          '日付':'datetime',
                                          '店舗ID':'store_id',
                                          '商品価格':'price',
                                          '売上個数':'sales'
                                          })

    # データの確認
    print(merged_df.dtypes)
    
    # datetime型への変換
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], format='%Y-%m-%d')
    
    # 西暦、月、日、曜日カラムの作成
    merged_df['year'] = merged_df['datetime'].dt.strftime("%Y")
    merged_df['month'] = merged_df['datetime'].dt.strftime("%m")
    
    '''
    月でビニング
    '''
    
    # 結合データの「年」と「月」をグルーピングして、「年」と「月」の全組み合わせを作成する
    time_group = merged_df.groupby(['year', 'month']).count().reset_index()[['year', 'month']]
    
    # カラム名「month_bining」として、通し番号をつける
    time_group['month_bining'] = list(range(len(time_group)))
    
    # testデータ用に、評価対象期間である2019年12月(month_bining: 23)のレコードを追加する
    time_group = time_group.append({'year':'2019', 'month':'12', 'month_bining':23}, ignore_index=True)
    
    # time_groupをCSVファイルに出力
    time_group.to_csv('./processed_data/time_group.csv', header=True, index=False)
    
    '''
    結合データに「month_bining」カラムを付与したデータフレームを作成（merged_df2）
    '''
    
    merged_df2 = pd.merge(merged_df, time_group, on=['year', 'month'], how='left')
    print(merged_df2.head())
    print(merged_df2.tail())
    
    '''
    月ごとの売上の集計(「商品ID」×「店舗ID」)
    '''
    
    # 「month_bining」「id」「store_id」でグルーピングし、売上個数の合計を取る
    monthly_sales = merged_df2.groupby(['month_bining', 'id', 'store_id']).agg({'sales': np.sum}).reset_index()
    print(monthly_sales.head())
    
    '''
    「'月'、'商品ID'、 '店舗ID'」の全組み合わせの作成
    '''
    
    # ライブラリのインポート
    from itertools import product
    
    # itertools.product関数を使ってすべての組み合わせをデータフレーム化
    all_combination = []
    for i in range(22):
        train_block = merged_df2[merged_df2['month_bining']==i]
        all_combination.append(np.array(list(product([i], train_block['id'].unique(), train_block['store_id'].unique()))))
        
    all_combination = pd.DataFrame(np.vstack(all_combination), columns=['month_bining', 'id', 'store_id'])
    all_combination.sort_values(['month_bining', 'id', 'store_id'], inplace=True)
    
    print(all_combination.info())
    
    '''
    all_combinationに月ごとの売上データ（monthly_sales）を結合
    '''
    
    pre_train = pd.merge(all_combination, monthly_sales, on=['month_bining', 'id', 'store_id'], how='left')
    
    # 売上個数がNaNとなっている箇所は、売上個数0個として値を埋める
    pre_train['sales'] = pre_train['sales'].fillna(0)
    
    '''
    pre_trainデータをCSVファイルとして出力
    '''
    
    pre_train.to_csv('./processed_data/pre_train.csv', header=True, index=False)


if __name__ == '__main__':
    train_data_processing()