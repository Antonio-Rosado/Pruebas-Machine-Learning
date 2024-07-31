import pandas as pd
import onnx
from pytorch_models import create_model_pytorch,compare_models_pytorch
from sklearn_models import create_model_sklearn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from ancillary_functions import add_lags, establish_target,split_train_test_at_point, load_model
import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random
from adapt.feature_based import CORAL
from adapt_models import adapt_model
from sklearn_models import sklearn_model, test_sklearn, compare_models_sklearn
from skorch import NeuralNetRegressor
from tensorflow_models import create_sequences, tensorflow_neural_network, test_tensorflow

df = pd.read_csv("output/" + "modified_m5_testing.csv")
lags = 5
features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']


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
df_train = df[df['d'] < validation_start].copy()
df_val = df[df['d'] >= validation_start].copy()


filename = 'xgb_one_item_400_2'
model_type = 'sk'
model_file = 'models/'+ filename + '.pickle'
X_train1, y_train1, X_test_final1, y_test_final1 = split_train_test_at_point(data_model,validation_start, features, target)
loaded_model = load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    X_train2, y_train2, X_test_final2, y_test_final2 = split_train_test_at_point(new_df, test_start, features, target)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = sklearn_model(new_df, X_train2, y_train2, X_test_final2, y_test_final2,validation_start,'Sales', features, target, 'xgb')
        mse1, mae1, mape1 = test_sklearn(loaded_model, new_df, validation_start, target, features, 'Sales')



        mse2, mae2, model = adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, 'coral',loaded_model,model_type)
        mse3, mae3, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'sa',loaded_model, model_type)
        mse4, mae4, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'bw',loaded_model, model_type)
        mse5, mae5, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'nnw',loaded_model, model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        maediff4 = mae4 - mae0
        msediff4 = mse4 - mse0
        maediff5 = mae5 - mae0
        msediff5 = mse5 - mse0
        results.append([idnames, mse0,mse1,mse2,mse3,mse4,mse5, mae0,mae1,mae2,mae3,mae4,mae5,msediff1,msediff2,msediff3,msediff4,msediff5,maediff1,maediff2,maediff3,maediff4,maediff5])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_sa','MSE_transfer_bw','MSE_transfer_nnw',
           'MAE_no_transfer','MAE_transfer_basic','MAE_transfer_coral', 'MAE_transfer_sa','MAE_transfer_bw','MAE_transfer_nnw',
           'MSE_diff_basic','MSE_transfer_coral', 'MSE_diff_sa','MSE_diff_bw','MSE_diff_nnw',
           'MAE_diff_basic','MAE_transfer_coral', 'MAE_diff_sa','MAE_diff_bw','MAE_diff_nnw']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")
print('done')


filename = 'xgb_many_item_400_2'
model_type = 'sk'
model_file = 'models/'+ filename + '.pickle'
X_train1, y_train1, X_test_final1, y_test_final1 = split_train_test_at_point(data_model,validation_start, features, target)
loaded_model = load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    X_train2, y_train2, X_test_final2, y_test_final2 = split_train_test_at_point(new_df, test_start, features, target)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = sklearn_model(new_df, X_train2, y_train2, X_test_final2, y_test_final2,validation_start,'Sales', features, target, 'xgb')
        mse1, mae1, mape1 = test_sklearn(loaded_model, new_df, validation_start, target, features, 'Sales')

        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'coral',
                                        loaded_model, model_type)
        mse3, mae3, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'sa',
                                        loaded_model, model_type)
        mse4, mae4, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'bw',
                                        loaded_model, model_type)
        mse5, mae5, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'nnw',
                                        loaded_model, model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        maediff4 = mae4 - mae0
        msediff4 = mse4 - mse0
        maediff5 = mae5 - mae0
        msediff5 = mse5 - mse0
        results.append(
            [idnames, mse0, mse1, mse2, mse3, mse4, mse5, mae0, mae1, mae2, mae3, mae4, mae5, msediff1, msediff2,
             msediff3, msediff4, msediff5, maediff1, maediff2, maediff3, maediff4, maediff5])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_sa',
               'MSE_transfer_bw', 'MSE_transfer_nnw',
               'MAE_no_transfer', 'MAE_transfer_basic', 'MAE_transfer_coral', 'MAE_transfer_sa', 'MAE_transfer_bw',
               'MAE_transfer_nnw',
               'MSE_diff_basic', 'MSE_transfer_coral', 'MSE_diff_sa', 'MSE_diff_bw', 'MSE_diff_nnw',
               'MAE_diff_basic', 'MAE_transfer_coral', 'MAE_diff_sa', 'MAE_diff_bw', 'MAE_diff_nnw']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")



filename = 'forest_one_item_400_2'
model_type = 'sk'
model_file = 'models/'+ filename + '.pickle'
X_train1, y_train1, X_test_final1, y_test_final1 = split_train_test_at_point(data_model,validation_start, features, target)
loaded_model = load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    X_train2, y_train2, X_test_final2, y_test_final2 = split_train_test_at_point(new_df, test_start, features, target)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = sklearn_model(new_df, X_train2, y_train2, X_test_final2, y_test_final2,validation_start,'Sales', features, target, 'forest')
        mse1, mae1, mape1 = test_sklearn(loaded_model, new_df, validation_start, target, features, 'Sales')


        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'coral',
                                        loaded_model, model_type)
        mse3, mae3, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'sa',
                                        loaded_model, model_type)
        mse4, mae4, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'bw',
                                        loaded_model, model_type)
        mse5, mae5, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'nnw',
                                        loaded_model, model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        maediff4 = mae4 - mae0
        msediff4 = mse4 - mse0
        maediff5 = mae5 - mae0
        msediff5 = mse5 - mse0
        results.append(
            [idnames, mse0, mse1, mse2, mse3, mse4, mse5, mae0, mae1, mae2, mae3, mae4, mae5, msediff1, msediff2,
             msediff3, msediff4, msediff5, maediff1, maediff2, maediff3, maediff4, maediff5])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_sa',
               'MSE_transfer_bw', 'MSE_transfer_nnw',
               'MAE_no_transfer', 'MAE_transfer_basic', 'MAE_transfer_coral', 'MAE_transfer_sa', 'MAE_transfer_bw',
               'MAE_transfer_nnw',
               'MSE_diff_basic', 'MSE_transfer_coral', 'MSE_diff_sa', 'MSE_diff_bw', 'MSE_diff_nnw',
               'MAE_diff_basic', 'MAE_transfer_coral', 'MAE_diff_sa', 'MAE_diff_bw', 'MAE_diff_nnw']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")


filename = 'forest_many_item_400_2'
model_type = 'sk'
model_file = 'models/'+ filename + '.pickle'
X_train1, y_train1, X_test_final1, y_test_final1 = split_train_test_at_point(data_model,validation_start, features, target)
loaded_model = load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    X_train2, y_train2, X_test_final2, y_test_final2 = split_train_test_at_point(new_df, test_start, features, target)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = sklearn_model(new_df, X_train2, y_train2, X_test_final2, y_test_final2,validation_start,'Sales', features, target, 'forest')
        mse1, mae1, mape1 = test_sklearn(loaded_model, new_df, validation_start, target, features, 'Sales')


        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'coral',
                                        loaded_model, model_type)
        mse3, mae3, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'sa',
                                        loaded_model, model_type)
        mse4, mae4, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'bw',
                                        loaded_model, model_type)
        mse5, mae5, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'nnw',
                                        loaded_model, model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        maediff4 = mae4 - mae0
        msediff4 = mse4 - mse0
        maediff5 = mae5 - mae0
        msediff5 = mse5 - mse0
        results.append(
            [idnames, mse0, mse1, mse2, mse3, mse4, mse5, mae0, mae1, mae2, mae3, mae4, mae5, msediff1, msediff2,
             msediff3, msediff4, msediff5, maediff1, maediff2, maediff3, maediff4, maediff5])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_sa',
               'MSE_transfer_bw', 'MSE_transfer_nnw',
               'MAE_no_transfer', 'MAE_transfer_basic', 'MAE_transfer_coral', 'MAE_transfer_sa', 'MAE_transfer_bw',
               'MAE_transfer_nnw',
               'MSE_diff_basic', 'MSE_transfer_coral', 'MSE_diff_sa', 'MSE_diff_bw', 'MSE_diff_nnw',
               'MAE_diff_basic', 'MAE_transfer_coral', 'MAE_diff_sa', 'MAE_diff_bw', 'MAE_diff_nnw']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")




filename = 'lstm_one_item_400_2'
model_type = 'tf'
model_file = 'models/'+ filename + '.keras'
X_train1, y_train1 = create_sequences(df_train, features, target, 30)
X_test_final1, y_test_final1 = create_sequences(df_val, features, target, 30)
X_train1 = X_train1.reshape(X_train1.shape[0], X_train1.shape[1] * X_train1.shape[2])
y_train1 = y_train1.reshape((y_train1.shape[0], y_train1.shape[1]))
X_test_final1 = X_test_final1.reshape(X_test_final1.shape[0], X_test_final1.shape[1] * X_test_final1.shape[2])
loaded_model = keras.models.load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    df_train2 = new_df[new_df['d'] < validation_start].copy()
    df_val2 = new_df[new_df['d'] >= validation_start].copy()
    X_train2, y_train2 = create_sequences(df_train2, features, target, 30)
    X_test_final2, y_test_final2 = create_sequences(df_val2, features, target, 30)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = tensorflow_neural_network(df_train2,df_val2,target,features, 4, 30, 'lstm', 50)

        mse1, mae1, mape1 = test_tensorflow(loaded_model, new_df,validation_start,30, target, features)

        print(X_train1.shape)

        X_train2 = X_train2.reshape(X_train2.shape[0], X_train2.shape[1] * X_train2.shape[2])
        X_test_final2 = X_test_final2.reshape(X_test_final2.shape[0], X_test_final2.shape[1] * X_test_final2.shape[2])
        y_train2 = y_train2.reshape((y_train2.shape[0], y_train2.shape[1]))


        mse2, mae2, model = adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, 'coral',loaded_model,model_type)
        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'sa',loaded_model, model_type)
        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'bw',loaded_model, model_type)
        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'nnw',loaded_model, model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        maediff4 = mae4 - mae0
        msediff4 = mse4 - mse0
        maediff5 = mae5 - mae0
        msediff5 = mse5 - mse0
        results.append([idnames, mse0, mse1, mse2, mse3, mse4, mse5, mae0, mae1, mae2, mae3, mae4, mae5, msediff1, msediff2,msediff3, msediff4, msediff5, maediff1, maediff2, maediff3, maediff4, maediff5])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_sa','MSE_transfer_bw', 'MSE_transfer_nnw',
               'MAE_no_transfer', 'MAE_transfer_basic', 'MAE_transfer_coral', 'MAE_transfer_sa', 'MAE_transfer_bw','MAE_transfer_nnw',
               'MSE_diff_basic', 'MSE_transfer_coral', 'MSE_diff_sa', 'MSE_diff_bw', 'MSE_diff_nnw',
               'MAE_diff_basic', 'MAE_transfer_coral', 'MAE_diff_sa', 'MAE_diff_bw', 'MAE_diff_nnw']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")


filename = 'lstm_many_item_400_2'
model_type = 'tf'
model_file = 'models/'+ filename + '.keras'
X_train1, y_train1 = create_sequences(df_train, features, target, 30)
X_test_final1, y_test_final1 = create_sequences(df_val, features, target, 30)
X_train1 = X_train1.reshape(X_train1.shape[0], X_train1.shape[1] * X_train1.shape[2])
y_train1 = y_train1.reshape((y_train1.shape[0], y_train1.shape[1]))
X_test_final1 = X_test_final1.reshape(X_test_final1.shape[0], X_test_final1.shape[1] * X_test_final1.shape[2])
loaded_model = keras.models.load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    df_train2 = new_df[new_df['d'] < validation_start].copy()
    df_val2 = new_df[new_df['d'] >= validation_start].copy()
    X_train2, y_train2 = create_sequences(df_train2, features, target, 30)
    X_test_final2, y_test_final2 = create_sequences(df_val2, features, target, 30)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = tensorflow_neural_network(df_train2,df_val2,target,features, 4, 30, 'lstm', 50)

        mse1, mae1, mape1 = test_tensorflow(loaded_model, new_df,validation_start,30, target, features)


        X_train2 = X_train2.reshape(X_train2.shape[0], X_train2.shape[1] * X_train2.shape[2])
        X_test_final2 = X_test_final2.reshape(X_test_final2.shape[0], X_test_final2.shape[1] * X_test_final2.shape[2])
        y_train2 = y_train2.reshape((y_train2.shape[0], y_train2.shape[1]))

        mse2, mae2, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'coral',
                                        loaded_model, model_type)
        mse3, mae3, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'sa',
                                        loaded_model, model_type)
        mse4, mae4, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'bw',
                                        loaded_model, model_type)
        mse5, mae5, model = adapt_model(X_train1, y_train1, X_train2, y_train2, X_test_final2, y_test_final2, 'nnw',
                                        loaded_model, model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        maediff3 = mae3 - mae0
        msediff3 = mse3 - mse0
        maediff4 = mae4 - mae0
        msediff4 = mse4 - mse0
        maediff5 = mae5 - mae0
        msediff5 = mse5 - mse0
        results.append(
            [idnames, mse0, mse1, mse2, mse3, mse4, mse5, mae0, mae1, mae2, mae3, mae4, mae5, msediff1, msediff2,
             msediff3, msediff4, msediff5, maediff1, maediff2, maediff3, maediff4, maediff5])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_coral', 'MSE_transfer_sa',
               'MSE_transfer_bw', 'MSE_transfer_nnw',
               'MAE_no_transfer', 'MAE_transfer_basic', 'MAE_transfer_coral', 'MAE_transfer_sa', 'MAE_transfer_bw',
               'MAE_transfer_nnw',
               'MSE_diff_basic', 'MSE_transfer_coral', 'MSE_diff_sa', 'MSE_diff_bw', 'MSE_diff_nnw',
               'MAE_diff_basic', 'MAE_transfer_coral', 'MAE_diff_sa', 'MAE_diff_bw', 'MAE_diff_nnw']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")

'''
filename = 'cnn_one_item_400_2'
model_type = 'tf'
model_file = 'models/'+ filename + '.keras'
X_train1, y_train1 = create_sequences(df_train, features, target, 30)
X_test_final1, y_test_final1 = create_sequences(df_val, features, target, 30)
loaded_model = keras.models.load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    df_train2 = new_df[new_df['d'] < validation_start].copy()
    df_val2 = new_df[new_df['d'] >= validation_start].copy()
    X_train2, y_train2 = create_sequences(df_train2, features, target, 30)
    X_test_final2, y_test_final2 = create_sequences(df_val2, features, target, 30)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = tensorflow_neural_network(df_train2,df_val2,target,features, 4, 30, 'cnn', 50)

        mse1, mae1, mape1 = test_tensorflow(loaded_model, new_df,validation_start,30, target, features)

        y_train1 = y_train1.reshape((y_train1.shape[0], y_train1.shape[1], 1))
        y_train2 = y_train2.reshape((y_train2.shape[0], y_train2.shape[1], 1))


        mse2, mae2, model = adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, 'rt',loaded_model,model_type)


        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        results.append([idnames, mse0,mse1,mse2, mae0,mae1, mae2,msediff1,msediff2,maediff1,maediff2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_adapt', 'MAE_no_transfer','MAE_transfer_basic','MAE_transfer_adapt',
           'MSE_diff_basic','MSE_diff_adapt','MAE_diff_basic','MAE_diff_adapt']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")


filename = 'cnn_many_item_400_2'
model_type = 'tf'
model_file = 'models/'+ filename + '.keras'
X_train1, y_train1 = create_sequences(df_train, features, target, 30)
X_test_final1, y_test_final1 = create_sequences(df_val, features, target, 30)
loaded_model = keras.models.load_model(model_file)
results=[]
for ids in idList:
    idnames = str(ids[0]) + '-' + str(ids[-1])

    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    df_train2 = new_df[new_df['d'] < validation_start].copy()
    df_val2 = new_df[new_df['d'] >= validation_start].copy()
    X_train2, y_train2 = create_sequences(df_train2, features, target, 30)
    X_test_final2, y_test_final2 = create_sequences(df_val2, features, target, 30)
    if ((new_df['Sales']!= 0).any()):
        mse0, mae0, model, mape0 = tensorflow_neural_network(df_train2,df_val2,target,features, 4, 30, 'cnn', 50)

        mse1, mae1, mape1 = test_tensorflow(loaded_model, new_df,validation_start,30, target, features)

        y_train1 = y_train1.reshape((y_train1.shape[0], y_train1.shape[1], 1))
        y_train2 = y_train2.reshape((y_train2.shape[0], y_train2.shape[1], 1))


        mse2, mae2, model = adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, 'rt',loaded_model,model_type)

        maediff1 = mae1 - mae0
        msediff1 = mse1 - mse0
        maediff2 = mae2 - mae0
        msediff2 = mse2 - mse0
        results.append([idnames, mse0,mse1,mse2, mae0,mae1, mae2,msediff1,msediff2,maediff1,maediff2])

columns = ['productIds', 'MSE_no_transfer', 'MSE_transfer_basic', 'MSE_transfer_adapt', 'MAE_no_transfer','MAE_transfer_basic','MAE_transfer_adapt',
           'MSE_diff_basic','MSE_diff_adapt','MAE_diff_basic','MAE_diff_adapt']
df1 = pd.DataFrame(results, columns=columns)
df1.set_index(['productIds'], inplace=True)
print(df1)
df1.to_excel("output/" + 'adapt_' + filename + ".xlsx")
'''
