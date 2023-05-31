import pandas as pd
#arima, autoarima, decision tree, random forest, adaboost, gaussian process, neural network
#tabla modelo-parámetros-mae-entrada-salida
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from preprocessing import *






def split_search_test(data,model,outputname, lags, steps, parameters):

    data_splines = add_splines(data,12,144)
    formatted_data = window_input_output(lags, steps, data, outputname)
    X_train, y_train, X_test_hyper, y_test_hyper, X_test_input, y_test_input, X_test_final, y_test_final = split_train_test(formatted_data, outputname)
    X_test_final_no_norm = X_test_final
    test_final_no_norm = y_test_final
    scaler = preprocessing.MinMaxScaler()
    if ((type(model).__name__)) == 'MLPRegressor':
        X_train = scaler.fit_transform(X_train)
        y_train = scaler.fit_transform(y_train)
        X_test_hyper = scaler.fit_transform(X_test_hyper)
        y_test_hyper = scaler.fit_transform(y_test_hyper)
        X_test_input = scaler.fit_transform(X_test_input)
        X_test_final = scaler.fit_transform(X_test_final)
        y_test_input = scaler.fit_transform(y_test_input)
        y_test_final = scaler.fit_transform(y_test_final)
    best_model, best_params = grid_search(X_train, y_train, X_test_hyper, y_test_hyper, model, parameters )
    if ((type(model).__name__)) == 'MLPRegressor':
        predictions_input = scaler.inverse_transform(best_model.predict(X_test_input))
        predictions_final = scaler.inverse_transform(best_model.predict(X_test_final))
        test_final = scaler.inverse_transform(y_test_final)
        test_input = scaler.inverse_transform(y_test_input)
    else:
        predictions_input = best_model.predict(X_test_input)
        predictions_final = best_model.predict(X_test_final)
        test_final = y_test_final
        test_input = y_test_input

    mae = mean_absolute_error(test_input, predictions_input)
    test_mae = mean_absolute_error(test_final, predictions_final)
    prediction_df = pd.DataFrame(predictions_final, index=X_test_final_no_norm.index)
    test_df = pd.DataFrame(test_final, index=test_final_no_norm.index)
    mae_by_day,mae_by_hour = get_mae_by_weekday(test_df,prediction_df)
    test_mse = mean_squared_error(test_final, predictions_final)
    test_mape = mean_absolute_percentage_error(test_final, predictions_final)
    return mae,best_params, test_mae, test_mse, test_mape, mae_by_day, mae_by_hour



def find_optimal_input(data,model,outputname, maxlags, steps, parameters):
    min_mae, params, test_mae, test_mse, test_mape, mae_day, mae_hour = split_search_test(data,model,outputname, 0, steps, parameters)
    lags = 0
    for i in range(1,maxlags+1):
        new_mae, new_params, new_test_mae, new_test_mse, new_test_mape, new_mae_day, new_mae_hour = split_search_test(data,model,outputname, i, steps, parameters)
        if (new_mae<min_mae):
            min_mae = new_mae
            params = new_params
            lags = i
            test_mae = new_test_mae
            test_mse = new_test_mse
            test_mape = new_test_mape
    return lags,params, test_mae, test_mse, test_mape, mae_day, mae_hour

def get_mae_by_weekday(test, predictions):

    mae_by_day = test.groupby(test.index.dayofweek).apply(lambda df: mean_absolute_error(df,predictions[predictions.index.dayofweek==df.index.dayofweek[0]]))
    mae_by_hour = test.groupby(test.index.hour).apply(lambda df: mean_absolute_error(df, predictions[predictions.index.hour == df.index.hour[0]]))

    return mae_by_day, mae_by_hour



def obtain_results_tables(data, config, outputname):
    results1 = []
    results2 = []
    for i in range(0,len(config)):
        model,model_params,maxlags,steps = build_model(config.iloc[i])
        print(model_params)
        model.random_state = 11
        output = outputname
        lags, best_params, mae,mse,mape, mae_day, mae_hour  = find_optimal_input(data, model, output, maxlags, steps,model_params)
        if (type(model).__name__=='RegressorChain'):
            name = type(model.base_estimator).__name__
        else:
            name = type(model).__name__
        results1.append([name,lags, steps, mae,mse,mape])
        results2.append([name, best_params, lags, steps])
        print(name)
        print(best_params)
        print(mae)
        print(mae_day)
        print(mae_hour)
    df1 = pd.DataFrame(results1, columns=['model', 'lags_used','steps_forecasted','mae','mse','mape'])
    df2 = pd.DataFrame(results2, columns=['model', 'parameters', 'lags_used', 'steps_forecasted'])
    print(df1)
    print(df2)
    df1.to_excel("tabla_resultados_fe.xlsx")
    df2.to_excel("tabla_parametros_fe.xlsx")




def grid_search(X_train, y_train, X_test, y_test, model, parameters):
    gs = GridSearchCV(model, parameters)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_







