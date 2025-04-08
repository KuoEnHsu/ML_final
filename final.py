import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

# 讀取資料
train_data = pd.read_csv('stage1data/preprocessed_train_features.csv')
train_data_target = pd.read_csv('stage1data/preprocessed_train_target.csv')
test_data = pd.read_csv('stage1data/preprocessed_test_features.csv')
test_data_2024=pd.read_csv('stage2data/preprocessed_test_features.csv')

#print(train_data_target)

#train_data=train_data.drop(train_data.columns[0],axis=1)


y=train_data_target.to_numpy()
X=train_data.to_numpy()
#X_test=test_data.to_numpy()
X_test_2024=test_data_2024.to_numpy()
#print(y)

X=np.delete(X,[0,1,2,4,5,10,37,38],axis=1)###1245
y = y.ravel()
#X_test=np.delete(X_test,[0,10,37,38],axis=1)
X_test_2024=np.delete(X_test_2024,[0,1,2,4,5,10,37,38],axis=1)
#print(X_test)
#print(X)
#print(y)

#final_model=RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=10)
#final_model.fit(X,y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=33)
# 建立 rf 模型
rfmodel=RandomForestClassifier(n_estimators=1000,n_jobs=-1,max_depth=4,max_features='log2',min_samples_leaf=1,min_samples_split=2,random_state=10)
# 使用訓練資料訓練模型
rfmodel.fit(X_train,y_train)
# 使用訓練資料預測分類
predicted = rfmodel.predict(X_val)
score=rfmodel.score(X_val,y_val)
print(score)
#### stage 1

y_test_predict=final_model.predict(X_test)

print(len(y_test_predict))

# 替換數據：1 -> 'true', 0 -> 'false'
data_replaced=np.where(y_test_predict==1, 'true' ,'false')

# 假設 ID 為連續整數（從 1 開始）
ids = np.arange(0, len(y_test_predict) )
# 建立 DataFrame
df = pd.DataFrame({'id': ids, 'home_team_win': data_replaced})

# 將 DataFrame 保存為 CSV 檔案
output_path = 'new_same_season_output.csv'
df.to_csv(output_path, index=False)


#### stage 2

final_model = RandomForestClassifier(n_estimators=1000,n_jobs=-1,max_depth=4,max_features='log2',min_samples_leaf=1,min_samples_split=2,random_state=10)
final_model.fit(X, y)

y_test_predict=final_model.predict(X_test_2024)

# 替換數據：1 -> 'true', 0 -> 'false'
data_replaced=np.where(y_test_predict==1, 'true' ,'false')

# 假設 ID 為連續整數（從 1 開始）
ids = np.arange(0, len(y_test_predict) )
# 建立 DataFrame
df = pd.DataFrame({'id': ids, 'home_team_win': data_replaced})

# 將 DataFrame 保存為 CSV 檔案
output_path = 'new_2024_output.csv'
df.to_csv(output_path, index=False)
