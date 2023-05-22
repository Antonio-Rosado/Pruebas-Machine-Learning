import pandas as pd
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




def window_input_output(input_length: int, output_length: int, data: pd.DataFrame, outputname: str) -> pd.DataFrame:
    df = data.copy()

    i = 1
    while i < input_length:
        df[f'x_{i}'] = df[outputname].shift(-i)
        i = i + 1

    j = 0
    while j < output_length:
        df[f'y_{j}'] = df[outputname].shift(-output_length - j)
        j = j + 1

    df = df.dropna(axis=0)

    return df



def split_train_test(data,outputname):
    size = int(len(data) * 0.60)
    train, test = data[0:size], data[size:len(data)]
    size2 = int(len(test) * 0.60)
    testsearch, testfinal = test[0:size2], test[size2:len(test)]
    size3 = int(len(testsearch) * 0.50)
    testhyper, testinput = testsearch[0:size3], testsearch[size3:len(testsearch)]
    X_cols = [col for col in data.columns if col.startswith('x')]
    X_cols.insert(0, outputname)
    y_cols = [col for col in data.columns if col.startswith('y')]
    X_train = train[X_cols].values
    y_train = train[y_cols].values
    X_test_hyper = testhyper[X_cols].values
    y_test_hyper = testhyper[y_cols].values
    X_test_input = testinput[X_cols].values
    y_test_input = testinput[y_cols].values
    X_test_final = testfinal[X_cols].values
    y_test_final = testfinal[y_cols].values

    return X_train, y_train, X_test_hyper, y_test_hyper, X_test_input, y_test_input, X_test_final, y_test_final


def select_model(model_name):
    if (model_name=='tree'):
        model = tree.DecisionTreeRegressor()
    if (model_name == 'forest'):
        model = RandomForestRegressor(n_jobs=-1)
    if (model_name=='adaboost'):
        adaboost_base = AdaBoostRegressor()
        model = RegressorChain(adaboost_base)
    if (model_name=='mlp'):
        model = MLPRegressor()
    if (model_name=='xgb'):
        model = xgb.XGBRegressor()
    if (model_name=='lgbm'):
        lgbm_base = lgb.LGBMRegressor()
        model = RegressorChain(lgbm_base)
    return model

def build_model(config):
    model = select_model(config[0])
    parameters = ast.literal_eval(config[1])
    lags = config[2]
    steps = config[3]
    return model,parameters,lags, steps













