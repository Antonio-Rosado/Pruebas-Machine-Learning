import pandas as pd
from pytorch_models import create_model_pytorch
from sklearn_models import create_model_sklearn
from ancillary_functions import add_lags, establish_target
import time
import random
from tensorflow_models import create_model_tensorflow



df = pd.read_csv("output/" + "modified_m5_testing.csv")
lags = 5
features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']

print(df)

forecast_lead = 30
outputname = 'Sales'

df, target = establish_target(df,outputname,forecast_lead)

trainingsize = 400

df = add_lags(df,lags,features)


test_start = 900
validation_start = 1400
train_limit= trainingsize


data_model = df[df['id'] < train_limit].copy()
idlist = random.sample(range(0,30489+1), trainingsize)
data_model2 = df[df['id'].isin(idlist)].copy()


results=[]

'''
model_file = 'models/cnn_one_item_400_2.keras'
st = time.time()
mse, mae, model,mape = create_model_tensorflow(data_model, test_start, validation_start, features, 4, 30, 'cnn', model_file,target, 50)
et = time.time()
train_time = et - st
results.append(['CNN_1', mse, mae, mape, train_time])
'''

model_file = 'models/xgb_one_item_400_2.pickle'
st = time.time()
mse, mae, model, mape = create_model_sklearn(data_model, test_start,validation_start, 'Sales', features,'xgb', model_file,target)
et = time.time()
train_time = et - st
results.append(['XGB_1', mse, mae, mape, train_time])


model_file = 'models/lstm_one_item_400_2.keras'
st = time.time()
mse, mae, model,mape = create_model_tensorflow(data_model, test_start, validation_start, features, 4, 30, 'lstm', model_file,target, 50)
et = time.time()
train_time = et - st
results.append(['LSTM_1', mse, mae, mape, train_time])


model_file = 'models/forest_one_item_400_2.pickle'
st = time.time()
mse, mae, model, mape = create_model_sklearn(data_model, test_start,validation_start, 'Sales', features,'forest', model_file,target)
et = time.time()
train_time = et - st
results.append(['FOREST_1', mse, mae, mape, train_time])

'''
model_file = 'models/cnn_many_item_400_2.keras'
st = time.time()
mse, mae, model,mape = create_model_tensorflow(data_model2, test_start, validation_start, features, 4, 30, 'cnn', model_file,target, 50)
et = time.time()
train_time = et - st
results.append(['CNN_MANY', mse, mae, mape, train_time])
'''

model_file = 'models/xgb_many_item_400_2.pickle'
st = time.time()
mse, mae, model, mape = create_model_sklearn(data_model2, test_start,validation_start, 'Sales', features,'xgb', model_file,target)
et = time.time()
train_time = et - st
results.append(['XGB_MANY', mse, mae, mape, train_time])


model_file = 'models/lstm_many_item_400_2.keras'
st = time.time()
mse, mae, model,mape = create_model_tensorflow(data_model2, test_start, validation_start, features, 4, 30, 'lstm', model_file,target, 50)
et = time.time()
train_time = et - st
results.append(['LSTM_MANY', mse, mae, mape, train_time])


model_file = 'models/forest_many_item_400_2.pickle'
st = time.time()
mse, mae, model, mape = create_model_sklearn(data_model2, test_start,validation_start, 'Sales', features,'forest', model_file,target)
et = time.time()
train_time = et - st
results.append(['FOREST_MANY', mse, mae, mape, train_time])



columns = ['model', 'MSE', 'MAE', 'MAPE','Train Time']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['model'], inplace=True)
print(df1)
df1.to_excel("output/" + "table_models_400_2.xlsx")






