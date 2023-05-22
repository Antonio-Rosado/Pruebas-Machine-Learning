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

    formatted_data = window_input_output(lags, steps, data, outputname)
    X_train, y_train, X_test_hyper, y_test_hyper, X_test_input, y_test_input, X_test_final, y_test_final = split_train_test(formatted_data, outputname)
    if ((type(model).__name__)) == 'MLPRegressor':
        X_train = preprocessing.normalize(X_train, norm='l2')
        y_train = preprocessing.normalize(y_train, norm='l2')
        X_test_hyper = preprocessing.normalize(X_test_hyper, norm='l2')
        y_test_hyper = preprocessing.normalize(y_test_hyper, norm='l2')
        X_test_input = preprocessing.normalize(X_test_input, norm='l2')
        y_test_input = preprocessing.normalize(y_test_input, norm='l2')
        X_test_final = preprocessing.normalize(X_test_final, norm='l2')
        y_test_final = preprocessing.normalize(y_test_final, norm='l2')
    best_model, best_params = grid_search(X_train, y_train, X_test_hyper, y_test_hyper, model, parameters )
    mae = mean_absolute_error(y_test_input, best_model.predict(X_test_input))
    test_mae = mean_absolute_error(y_test_final, best_model.predict(X_test_final))
    test_mse = mean_squared_error(y_test_final, best_model.predict(X_test_final))
    test_mape = mean_absolute_percentage_error(y_test_final, best_model.predict(X_test_final))
    return mae,best_params, test_mae, test_mse, test_mape



def find_optimal_input(data,model,outputname, maxlags, steps, parameters):
    min_mae, params, test_mae, test_mse, test_mape = split_search_test(data,model,outputname, 0, steps, parameters)
    lags = 0
    for i in range(1,maxlags+1):
        new_mae, new_params, new_test_mae, new_test_mse, new_test_mape = split_search_test(data,model,outputname, i, steps, parameters)
        if (new_mae<min_mae):
            min_mae = new_mae
            params = new_params
            lags = i
            test_mae = new_test_mae
            test_mse = new_test_mse
            test_mape = new_test_mape
    return lags,params, test_mae, test_mse, test_mape

def obtain_results_tables(data, config, outputname):
    results1 = []
    results2 = []
    for i in range(0,len(config)):
        model,model_params,maxlags,steps = build_model(config.iloc[i])
        print(model_params)
        model.random_state = 11
        output = outputname
        lags, best_params, mae,mse,mape  = find_optimal_input(data, model, output, maxlags, steps,model_params)
        if (type(model).__name__=='RegressorChain'):
            name = type(model.base_estimator).__name__
        else:
            name = type(model).__name__
        results1.append([name,lags, steps, mae,mse,mape])
        results2.append([name, best_params, lags, steps])
        print(name)
        print(best_params)
        print(mae)
    df1 = pd.DataFrame(results1, columns=['model', 'lags_used','steps_forecasted','mae','mse','mape'])
    df2 = pd.DataFrame(results2, columns=['model', 'parameters', 'lags_used', 'steps_forecasted'])
    print(df1)
    print(df2)
    df1.to_excel("tabla_resultados.xlsx")
    df2.to_excel("tabla_parametros.xlsx")



def grid_search(X_train, y_train, X_test, y_test, model, parameters):
    gs = GridSearchCV(model, parameters)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_







