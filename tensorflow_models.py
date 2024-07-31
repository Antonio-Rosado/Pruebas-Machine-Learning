import math
from ancillary_functions import save_model, split_train_test_at_point
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from adapt.feature_based import CORAL
from adapt.parameter_based import FineTuning
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, MaxPooling2D, Reshape
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from skorch import NeuralNetRegressor, NeuralNet
from sklearn_models import skorch_model



def create_sequences(data, features, target, seq_length):
    X_data = data[features]
    y_data = data[target]
    X = []
    y = []
    for i in range(0,len(data) - seq_length,seq_length):
        X.append(X_data[:][i:i+seq_length])
        y.append(y_data[:][i:i+seq_length])
    return np.array(X), np.array(y)


def tensorflow_neural_network(df_train,df_test,target,features, batch_size, sequence_length, n_type, epochs):


    for c in df_train.columns:
        mean = df_train[target].mean()
        stdev = df_train[target].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev


    sequence_length = sequence_length
    batch_size = batch_size

    X_train, y_train = create_sequences(df_train, features, target, sequence_length)

    X_test, y_test = create_sequences(df_test, features, target, sequence_length)

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    ogshape = X_train.shape

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])

    model = select_network(n_type,X_train, ogshape)

    model.compile(loss='mean_squared_error', optimizer='adam')
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
    print(y_train.shape)
    print(X_train.shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    predictions = model.predict(X_test)
    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]))


    mse = mean_squared_error(predictions, y_test)
    print(mse)
    mae = mean_absolute_error(predictions, y_test)
    print(mae)
    mape = mean_absolute_percentage_error(predictions, y_test)
    print(mape)

    return mse, mae, model, mape


def select_network(nn_type, X_train, ogshape):
    if (nn_type == 'lstm'):
        model = Sequential()
        model.add(Reshape((ogshape[1], ogshape[2])))
        model.add(LSTM(units=32, return_sequences=True, input_shape=(ogshape[1], ogshape[2])))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(LSTM(units=16, return_sequences=True))
        model.add(Dense(units=1))

    if (nn_type == 'cnn'):
        model = Sequential()
        model.add(Conv1D(64, 1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
    return model


def test_tensorflow(model, data,test_start,sequence_length, target, features):


    df_test = data[data['d'] >= test_start].copy()
    df = df_test.copy()

    for c in df.columns:
        mean = df[target].mean()
        stdev = df[target].std()

        df[c] = (df[c] - mean) / stdev

    X_test, y_test = create_sequences(df, features, target, sequence_length)

    predictions = model.predict(X_test)

    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]))

    mse = mean_squared_error(predictions, y_test)
    print(mse)
    mae = mean_absolute_error(predictions, y_test)
    print(mae)
    mape = mean_absolute_percentage_error(predictions, y_test)
    print(mape)

    return mse,mae,mape



def create_model_tensorflow(data, test_start,validation_start, features,batch_size,sequence_length, nn_type, model_file, target, epochs):

    df_train = data[data['d'] < test_start].copy()
    df_t = data[data['d'] >= test_start].copy()
    df_test = df_t[df_t['d'] < validation_start].copy()
    mse, mae, model,mape = tensorflow_neural_network(df_train,df_test,target,features, batch_size, sequence_length, nn_type, epochs)

    model.save(model_file)

    return mse, mae, model,mape


def compare_models_tensorflow(df_transfer,features,test_start,validation_start, outputname,loaded_model, productIds,batch_size, sequence_length, nntype, target, forecast_lead, epochs):


    new_df = df_transfer[df_transfer['id'].isin(productIds)].copy()

    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)

    new_df = new_df.dropna(subset=[target])


    pd.set_option('display.max_columns', None)
    print(new_df)

    mse1, mae1, model,mape = tensorflow_neural_network(new_df,target,features,test_start,validation_start, batch_size, sequence_length, nntype, epochs)

    mse2, mae2, mape2 = test_tensorflow(loaded_model, new_df, validation_start, sequence_length, target, features)

    print('MSE without tansfer:' + str(mse1))

    print('MSE with tansfer:' + str(mse2))

    return mse1, mse2, mae1, mae2


