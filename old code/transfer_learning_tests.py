import pandas as pd
from tranfer_learning import create_model_pytorch,test_pytorch, create_model_sklearn, compare_models_pytorch, compare_models_sklearn
import time
import random
from pytorch import load_model



df = pd.read_csv("output/" + "modified_m5.csv")
lags = 5
features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']

print(df)



for i in range(0,lags):
    df['lag_sales_' + str(i)] = df.groupby('id')['Sales'].shift(i)
    features.append('lag_sales_' + str(i))

df = df.fillna(method='bfill')


test_start = 900
validation_start = 1400
train_limit= 100


idList = [range(101,111),range(111,121),range(121,131),range(131,141),range(141,151),range(151,161),range(161,171),range(171,181),range(181,191),range(191,201)]
df_transfer = df.copy()

results=[]



filename = 'forest_many_item'

model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)



for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    #mse1, mse2, mae1, mae2 =  compare_models_pytorch(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 4, 30, 'lstm')
    mse1, mse2, mae1, mae2 = compare_models_sklearn(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, i, 'forest')
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + ".xlsx")





