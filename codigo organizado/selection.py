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






def split_search_test(data,model,outputname, lags, steps, parameters, fe):

    if(fe==True):
        data = feature_engineering(data)

    formatted_data = window_input_output(lags, steps, data, outputname)
    X_train, y_train, X_test_hyper, y_test_hyper, X_test_input, y_test_input, X_test_final, y_test_final = split_train_test(formatted_data, outputname)
    X_test_final_no_norm = X_test_final
    test_final_no_norm = y_test_final
    scaler_X = preprocessing.MinMaxScaler()
    scaler_y = preprocessing.MinMaxScaler()
    if ((type(model).__name__)) == 'MLPRegressor':
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train)
        X_test_hyper = scaler_X.transform(X_test_hyper)
        y_test_hyper = scaler_y.transform(y_test_hyper)
        X_test_input = scaler_X.transform(X_test_input)
        X_test_final = scaler_X.transform(X_test_final)
        y_test_input = scaler_y.transform(y_test_input)
        y_test_final = scaler_y.transform(y_test_final)
    best_model, best_params = grid_search(X_train, y_train, X_test_hyper, y_test_hyper, model, parameters )
    if ((type(model).__name__)) == 'MLPRegressor':
        predictions_input = scaler_y.inverse_transform(best_model.predict(X_test_input))
        predictions_final = scaler_y.inverse_transform(best_model.predict(X_test_final))
        test_final = scaler_y.inverse_transform(y_test_final)
        test_input = scaler_y.inverse_transform(y_test_input)
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



def find_optimal_input(data,model,outputname, maxlags, steps, parameters,fe):
    min_mae, params, test_mae, test_mse, test_mape, mae_day, mae_hour = split_search_test(data,model,outputname, 0, steps, parameters,fe)
    lags = 0
    for i in range(1,maxlags+1):
        new_mae, new_params, new_test_mae, new_test_mse, new_test_mape, new_mae_day, new_mae_hour = split_search_test(data,model,outputname, i, steps, parameters,fe)
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






def obtain_results_tables(data, config, outputname,fe, result_name):
    results1 = []
    results2 = []
    for i in range(0,len(config)):
        model,model_params,maxlags,steps = build_model(config.iloc[i])
        print(model_params)
        model.random_state = 11
        output = outputname
        lags, best_params, mae,mse,mape, mae_day, mae_hour  = find_optimal_input(data, model, output, maxlags, steps,model_params,fe)
        if (type(model).__name__=='RegressorChain'):
            name = type(model.base_estimator).__name__
        else:
            name = type(model).__name__
        results1.append([name,lags, steps, mae,mse,mape])
        results2.append([name, best_params, lags, steps])
        for day in mae_day:
            results1[i].append(day)
        for hour in mae_hour:
            results1[i].append(hour)
        print(results1)
        print(name)
        print(best_params)
        print(mae)
    columns1 = ['model', 'lags_used','steps_forecasted','mae','mse','mape']
    for i in range(0,7):
        columns1.append("mae_day_"+str(i))
    for i in range(0,24):
        columns1.append("mae_hour_"+str(i))
    print(columns1)
    df1 = pd.DataFrame(results1, columns=columns1)
    df2 = pd.DataFrame(results2, columns=['model', 'parameters', 'lags_used', 'steps_forecasted'])
    print(df1)
    print(df2)
    df1.to_excel(result_name + "_resultados.xlsx")
    df2.to_excel(result_name + "_parametros.xlsx")




def grid_search(X_train, y_train, X_test, y_test, model, parameters):
    gs = GridSearchCV(model, parameters)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def obtain_results_tables_tsfresh(data, config, outputname, result_name):

    X,y = extract_feautres_tsfresh(data, outputname)
    size = int(len(data) * 0.60)
    results1 = []
    results2 = []
    for i in range(0, len(config)):
        model, model_params, maxlags, steps = build_model(config.iloc[i])
        X_train, y_train, X_test_search, y_test_search, X_test_final, y_test_final = get_train_test_tsfresh(X, y, steps,size)

        print(model_params)
        model.random_state = 11
        output = outputname
        best_params, mae, mse, mape,mae_day,mae_hour = find_best_model_tsfresh(model,X_train,y_train,X_test_search,y_test_search,X_test_final,y_test_final,model_params)

        if (type(model).__name__ == 'RegressorChain'):
            name = type(model.base_estimator).__name__
        else:
            name = type(model).__name__
        results1.append([name, maxlags, steps, mae, mse, mape])
        results2.append([name, best_params, maxlags, steps])
        for day in mae_day:
            results1[i].append(day)
        for hour in mae_hour:
            results1[i].append(hour)

        print(results1)
        print(name)
        print(best_params)
        print(mae)
    columns1 = ['model', 'steps_forecasted', 'mae', 'mse', 'mape']
    for i in range(0,7):
        columns1.append("mae_day_"+str(i))
    for i in range(0,24):
        columns1.append("mae_hour_"+str(i))
    df1 = pd.DataFrame(results1, columns=columns1)
    df2 = pd.DataFrame(results2, columns=['model', 'parameters', 'steps_forecasted'])
    print(df1)
    print(df2)
    df1.to_excel(result_name + "_resultados_tsfresh.xlsx")
    df2.to_excel(result_name + "_parametros_tsfresh.xlsx")




def find_best_model_tsfresh(model,X_train,y_train,X_test_search,y_test_search,X_test_final,y_test_final, model_params):
    print(X_test_final)
    print(y_test_final)
    X_test_final_no_norm = X_test_final
    test_final_no_norm = y_test_final
    scaler_X = preprocessing.MinMaxScaler()
    scaler_y = preprocessing.MinMaxScaler()
    if ((type(model).__name__)) == 'MLPRegressor':
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train)
        X_test_search = scaler_X.transform(X_test_search)
        X_test_final = scaler_X.transform(X_test_final)
        y_test_search = scaler_y.transform(y_test_search)
        y_test_final = scaler_y.transform(y_test_final)
    best_model, best_params = grid_search(X_train, y_train, X_test_search, y_test_search, model, model_params)
    if ((type(model).__name__)) == 'MLPRegressor':
        predictions_final = scaler_y.inverse_transform(best_model.predict(X_test_final))
        test_final = scaler_y.inverse_transform(y_test_final)
    else:
        predictions_final = best_model.predict(X_test_final)
        test_final = y_test_final
    mae = mean_absolute_error(test_final, predictions_final)
    mse = mean_squared_error(test_final, predictions_final)
    prediction_df = pd.DataFrame(predictions_final, index=X_test_final_no_norm.index)
    test_df = pd.DataFrame(test_final_no_norm, index=y_test_final.index)
    mae_by_day, mae_by_hour = get_mae_by_weekday(test_df, prediction_df)
    mape = mean_absolute_percentage_error(test_final, predictions_final)
    return  best_params,mae,mse, mape, mae_by_day, mae_by_hour