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
from selection import obtain_results_tables
from selection import obtain_results_tables_tsfresh
from preprocessing import periodic_spline_transformer
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame

if __name__ == '__main__':
    data = pd.read_excel('Demanda_2015.xlsx', names=['DATE', 'TIME', 'DEMAND'])
    data['DATE-TIME'] = data.apply(lambda r : pd.datetime.combine(r['DATE'],r['TIME']),1)
    data = data.drop(columns=['DATE','TIME'])
    data = data[['DATE-TIME','DEMAND']]

    data = data.set_index('DATE-TIME')


    print(data.describe())
    print(data.head())
    print(data.tail())

    config = pd.read_excel('configuraciontest.xlsx')
    print(type(config.iloc[0][0]))
    
    """
    data['Name'] = 'A'
    data2 = data.set_index('Name')
    y = data2[['DEMAND']].squeeze()
    print(y)
    print(type(y))
    selected_features = extract_relevant_features(data, y,column_id='Name')

    print(selected_features.head())
    print(selected_features.tail())
    
    print(data)
    """

    #data['Name'] = 'A'
    #data2 = data.set_index('Name')
    '''
    df_shift, y = make_forecasting_frame(data['DEMAND'], kind='demand', max_timeshift=12, rolling_direction=1)
    df_shift.groupby("id").size().agg([np.min, np.max])
    print(y)
    print(df_shift)
    #df_shift = df_shift.set_index(id)
    #df_shift = df_shift.set_index(df_shift.index.map(lambda x: x[1]), drop=True)
    #print(df_shift)
    X = extract_relevant_features(df_shift, y, column_id="id", column_sort="time", column_value="value",
                                  default_fc_parameters=MinimalFCParameters())
    y2 = pd.DataFrame(y)
    j = 0
    while j < 6:
        y2[f'y_{j}'] = y2['value'].shift(-6 - j)
        j = j + 1

    y2 = y2.dropna(axis=0)
    y2 = y2[y2.index.isin(X.index)]
    X = X[X.index.isin(y2.index)]
    y2 = y2.set_index(y2.index.map(lambda x: x[1]), drop=True)
    y2.index.name = "id"
    print(X)
    print(X.index[0][1])
    print(y2)
    print(y2.index[0])
    print(y2.index.dayofweek[0])
    print(X.index[0][1].dayofweek)
    
    print(X)
    print(X.index[0])
    print(X.index.dayofweek[0])
    size = int(len(data) * 0.60)
    X_train, X_test = X[0:size], X[size:len(X)]
    y_train, y_test = y2[0:size], y2[size:len(y2)]
    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)
'''
    #obtain_results_tables(data,config,'DEMAND',fe=False, result_name = "no_fe")
    #obtain_results_tables(data, config, 'DEMAND', fe=True, result_name="fe")
    obtain_results_tables_tsfresh(data, config, 'DEMAND', result_name="tabla")




#limpiar código
#pasar configuración a archivo
#grid search (hiperparámetros modelo/prcesamiento) duración < 15 min
#excel resultados(subir a repositorio) :modelo, entradas, salidas, mae, mse, mape (encontrar mejor modelo) + tabla modelo-hiperparámetro-entrada-salida
#random forest: n_jobs = -1, mlp:datos normalizados
#xgboost, light gbm
#salida: 1 hora, 8 horas, 24 horas
# mae según mes, hora, día de la semana


