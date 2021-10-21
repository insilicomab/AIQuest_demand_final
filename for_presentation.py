# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

'''
データ加工
'''

# データの読み込み
test = pd.read_csv('test.csv')
category_names = pd.read_csv('category_names.csv')
item_categories = pd.read_csv('item_categories.csv')
actual = pd.read_csv('answer.csv', header=None)
pred = pd.read_csv('demand_RandomForest.csv', header=None)

# 列名の変更
actual = actual.rename(columns={0: 'index', 1: 'actual'})
pred = pred.rename(columns={0: 'index', 1: 'pred'})

# データの結合
test = pd.merge(test, item_categories, on='商品ID')
test = pd.merge(test, category_names, on='商品カテゴリID')
test = pd.merge(test, actual, on='index')
test = pd.merge(test, pred, on='index')

# 予測結果を四捨五入したカラムを作成
test['pred_round'] = np.round(test['pred'])

# 不要なカラムの削除
test = test.drop(['店舗ID', '商品カテゴリID'], axis=1)

# 誤差（実測値－予測値）の絶対値カラムを作成
test['diff'] = np.abs(test['actual']-test['pred_round'])

''''
評価関数
'''

# R2値
r2 = r2_score(test['actual'], test['pred'])
print(f'R2値 = {r2}')

# MAE
mae = mean_absolute_error(test['actual'], test['pred'])
print(f'MAE = {mae}')

# RMSE
rmse = np.sqrt(mean_squared_error(test['actual'], test['pred']))
print(f'RMSE = {rmse}')

'''
可視化(散布図)
'''

# 散布図の作成
plt.figure(figsize=(10,10))
sns.scatterplot(x='pred', y='actual', s=100, data=test)
plt.plot([0, 70], [0, 70], linestyle = "--", linewidth=4)
plt.xlabel('')
plt.ylabel('')
plt.show()

'''
可視化（棒グラフ）
'''

plt.figure(dpi=500)
plt.bar(test['index'], -test['actual'])
plt.bar(test['index'], test['pred'])
plt.ylim(-80, 70)
plt.yticks(())
plt.xticks(())
plt.gca().spines['right'].set_visible(False) # 右枠削除
plt.gca().spines['top'].set_visible(False) # 上枠削除
plt.show()

'''
可視化（折れ線グラフ）
'''

plt.figure(dpi=500)
sns.lineplot(x='index', y='actual', data=test)
sns.lineplot(x='index', y='pred', data=test)
plt.xticks(())
plt.xlabel('')
plt.ylabel('')
plt.gca().spines['right'].set_visible(False) # 右枠削除
plt.gca().spines['top'].set_visible(False) # 上枠削除
plt.show()

'''
誤差上位20のデータを抽出し、可視化
'''

diff_worst_20 = test.sort_values('diff', ascending=False).iloc[0:20,:]

plt.figure(dpi=500)
diff_worst_20.sort_values('diff', ascending=True).plot.barh(x='商品ID', y='diff')
plt.ylabel('')
plt.show()