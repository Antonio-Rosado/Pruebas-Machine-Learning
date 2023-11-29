import pandas as pd
#arima, autoarima, decision tree, random forest, adaboost, gaussian process, neural network
#tabla modelo-parámetros-mae-entrada-salida
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import xgboost as xgb
import lightgbm as lgb
from itertools import product
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
from selection import print_mse_mae_all
from pytorch import pytorch_lstm, pytorch_cnn, pytorch_transformer
from preprocessing import periodic_spline_transformer
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame

if __name__ == '__main__':

    #Demanda 2015
    data = pd.read_excel('datasets/Demanda_2015.xlsx', names=['DATE', 'TIME', 'DEMAND'])
    data['DATE-TIME'] = data.apply(lambda r : pd.datetime.combine(r['DATE'],r['TIME']),1)
    data = data.drop(columns=['DATE','TIME'])
    data = data[['DATE-TIME','DEMAND']]

    data = data.set_index('DATE-TIME')

    data['DEMAND2'] = data['DEMAND']

    forecast_lead = 30
    target = f"{'DEMAND'}_lead{forecast_lead}"
    features = list(data.columns.difference(['DEMAND']))
    print(features)

    data[target] = data['DEMAND'].shift(-forecast_lead)
    data = data.iloc[:-forecast_lead]
    test_start = "2015-10-10 00:00:00"

    #Desempleo EEUU
    #data = pd.read_excel('datasets/Desempleo EEUU.xls', names=['DATE', 'UNEMPLOYMENT CLAIMS'])
    #data['DATE'] = pd.to_datetime(data['DATE'])
    #data = data.set_index('DATE')
    #data['UNEMPLOYMENT CLAIMS 2'] = data['UNEMPLOYMENT CLAIMS']

    #forecast_lead = 30
    #target = f"{'UNEMPLOYMENT CLAIMS'}_lead{forecast_lead}"
    #features = list(data.columns.difference(['UNEMPLOYMENT CLAIMS']))
    #print(features)

    #data[target] = data['UNEMPLOYMENT CLAIMS'].shift(-forecast_lead)
    #data = data.iloc[:-forecast_lead]
    #test_start = "2005-01-01"

    # data[target] = data['DEMAND'].shift(-forecast_lead)
    # data = data.iloc[:-forecast_lead]
    # test_start = "2015-10-10 00:00:00"

    #data = pd.read_csv('datasets/Calidad Aire Milan.csv')
    #data['local_datetime'] = pd.to_datetime(data['local_datetime'])
    #data = data.set_index('local_datetime')



    print(data.describe())
    print(data.head())
    print(data.tail())

    config = pd.read_excel('configuraciontest.xlsx')
    print(type(config.iloc[0][0]))
    

    #obtain_results_tables(data,config,'pm2p5',fe=False, result_name = "no_fe",hour_week_data = True)
    #obtain_results_tables(data, config, 'pm2p5', fe=True, result_name="fe",hour_week_data = True)
    #obtain_results_tables_tsfresh(data, config, 'pm2p5', result_name="tabla",lags=15,hour_week_data = True)
    #pytorch_cnn(data, forecast_lead, target, features, test_start)
    #pytorch_lstm(data, forecast_lead, target, features, test_start)
    #pytorch_transformer(data, forecast_lead, target, features, test_start)
    #mse_basic(data, 'DEMAND', 6, 6, 'xgb')
    #mse_basic(data, 'DEMAND', 6, 6, 'forest')
    print_mse_mae_all(data, forecast_lead, target, features, test_start, 'DEMAND', 6, 6)


