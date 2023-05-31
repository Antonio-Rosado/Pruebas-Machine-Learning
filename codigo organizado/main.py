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
from preprocessing import periodic_spline_transformer


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

obtain_results_tables(data,config,'DEMAND')







#limpiar código
#pasar configuración a archivo
#grid search (hiperparámetros modelo/prcesamiento) duración < 15 min
#excel resultados(subir a repositorio) :modelo, entradas, salidas, mae, mse, mape (encontrar mejor modelo) + tabla modelo-hiperparámetro-entrada-salida
#random forest: n_jobs = -1, mlp:datos normalizados
#xgboost, light gbm
#salida: 1 hora, 8 horas, 24 horas
# mae según mes, hora, día de la semana


