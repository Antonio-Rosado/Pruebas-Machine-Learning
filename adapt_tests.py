import pandas as pd
import onnx
from pytorch_models import create_model_pytorch,compare_models_pytorch
from sklearn_models import create_model_sklearn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from ancillary_functions import add_lags, establish_target,split_train_test_at_point, load_model
import time
import random
from adapt.feature_based import CORAL
from adapt_models import adapt_model
from sklearn_models import sklearn_model, test_sklearn, compare_models_sklearn
from skorch import NeuralNetRegressor


df = pd.read_csv("output/" + "modified_m5_testing.csv")
lags = 5
features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']

print(df)

forecast_lead = 30
outputname = 'Sales'

df, target = establish_target(df,outputname,forecast_lead)


trainingsize = 200

df = add_lags(df,lags,features)
df_transfer = df.copy()

test_start = 900
validation_start = 1400
train_limit= trainingsize





idList = [range(101,121),range(121,141),range(141,161),range(161,181),range(181,201),range(201,221),range(221,241),range(241,261),range(261,281),range(281,301)]




data_model = df[df['id'] < train_limit].copy()
X_train1, y_train1, X_test_final1, y_test_final1 = split_train_test_at_point(data_model,validation_start, features, target)

filename = 'forest_many_item_400'
model_file = 'models/'+filename +'.pickle'
loaded_model = load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    X_train2, y_train2, X_test_final2, y_test_final2 = split_train_test_at_point(new_df, test_start, features, target)

    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = sklearn_model(new_df, X_train2, y_train2, X_test_final2, y_test_final2,
                                                 validation_start,
                                                 'Sales', features, target, 'forest')
        mse1, mae1, mape1 = test_sklearn(loaded_model, new_df, validation_start, target, features, 'Sales')
        mse2, mae2, model = adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, 'coral',loaded_model)
        mse3, mae3, model = adapt_model(X_train1, y_train1, X_train2,y_train2, X_test_final2, y_test_final2, 'r2',loaded_model)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        results.append([idnames, mse0,mse1,mse2,mse3, mae0,mae1, mae2,mae3,msediff1,maediff1,msediff2,maediff2,msediff3,maediff3])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_tradaboostr2', 'MAE_no_transfer','MAE_transfer_basic','MAE_transfer_coral','MAE_transfer_tradaboostr2',
           'MSE_diff_basic','MSE_diff_coral','MSE_diff_tradaboostr2','MAE_diff_basic','MAE_diff_coral','MAE_diff_tradaboostr2']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)










