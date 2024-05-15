import pandas as pd
from sklearn_models import compare_models_sklearn
from pytorch_models import compare_models_pytorch
from ancillary_functions import load_model, add_lags, establish_target
from adapt.feature_based import CORAL



df = pd.read_csv("output/" + "modified_m5_testing.csv")
lags = 5
features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']

df = add_lags(df,lags,features)
print(df)






test_start = 900
validation_start = 1400


idList = [range(101,121),range(121,141),range(141,161),range(161,181),range(181,201),range(201,221),range(221,241),range(241,261),range(261,281),range(281,301)]

forecast_lead = 30
outputname = 'Sales'

df,target = establish_target(df,outputname,forecast_lead)

df_transfer = df.copy()

filename2 = 'adapt'



results=[]
filename = 'forest_many_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)


for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    mse1, mse2, mae1, mae2 = compare_models_sklearn(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 'forest',target,forecast_lead)
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + filename2 + ".xlsx")

'''
results=[]
filename = 'forest_one_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)

for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    mse1, mse2, mae1, mae2 = compare_models_sklearn(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 'forest',target,forecast_lead)
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + filename2 + ".xlsx")


results=[]
filename = 'xgb_one_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)

for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    mse1, mse2, mae1, mae2 = compare_models_sklearn(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 'xgb',target,forecast_lead)
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + filename2 + ".xlsx")

results=[]
filename = 'xgb_many_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)

for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    mse1, mse2, mae1, mae2 = compare_models_sklearn(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 'xgb',target,forecast_lead)
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + filename2 + ".xlsx")


results=[]
filename = 'lstm_many_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)

for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    mse1, mse2, mae1, mae2 =  compare_models_pytorch(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 4, 30, 'lstm',target,forecast_lead)
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + filename2 + ".xlsx")


results=[]
filename = 'lstm_one_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)

for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])
    mse1, mse2, mae1, mae2 =  compare_models_pytorch(df_transfer,features,test_start,validation_start, 'Sales',loaded_model, ids, 4, 30, 'lstm',target,forecast_lead)
    results.append([idnames,mse1,mse2,mae1,mae2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer', 'MAE_no_transfer','MAE_transfer']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + filename + filename2 + ".xlsx")

'''






