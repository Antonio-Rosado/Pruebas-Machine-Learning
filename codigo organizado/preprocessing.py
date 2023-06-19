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
from sklearn.preprocessing import SplineTransformer
from sklearn import tree
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame




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
    X_train = train[X_cols]
    y_train = train[y_cols]
    X_test_hyper = testhyper[X_cols]
    y_test_hyper = testhyper[y_cols]
    X_test_input = testinput[X_cols]
    y_test_input = testinput[y_cols]
    X_test_final = testfinal[X_cols]
    y_test_final = testfinal[y_cols]

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


def add_time_features(data):
    data['hour'] = [data.index[i].hour for i in range(len(data))]
    data['month'] = [data.index[i].month for i in range(len(data))]
    data['dayofweek'] = [data.index[i].dayofweek for i in range(len(data))]
    data['dayofmonth'] = [data.index[i].day for i in range(len(data))]
    return data

def add_windows(data):
    load_val = data[['DEMAND']]
    window_rol = load_val.rolling(12, min_periods=1)
    data_rolling = pd.concat([window_rol.min(), window_rol.mean(), window_rol.max()], axis=1)
    data_rolling.columns = ['min_rol', 'mean_rol', 'max_rol']
    window_ex = load_val.expanding()
    data_expanding = pd.concat([window_ex.min(), window_ex.mean(), window_ex.max()], axis=1)
    data_expanding.columns = ['min_ex', 'mean_ex', 'max_ex']
    new_data = pd.concat([data, data_rolling, data_expanding], axis=1)
    return new_data

def add_splines(data,n_splines,period):
    splines = periodic_spline_transformer(period, n_splines=n_splines).fit_transform(data)
    splines_df = pd.DataFrame(splines, columns=[f"x_spline_{i}" for i in range(splines.shape[1])], )
    splines_df = splines_df.set_index(data.index)
    new_data = pd.concat([data,splines_df], axis = 1)
    return new_data


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )



def feature_engineering(data):
    data = add_splines(data, 12, 144)
    data = add_time_features(data)
    data = add_windows(data)
    return data


def extract_feautres_tsfresh(data,outputname):
    df_shift, y = make_forecasting_frame(data[outputname], kind=outputname, max_timeshift=12, rolling_direction=1)
    df_shift.groupby("id").size().agg([np.min, np.max])
    X = extract_relevant_features(df_shift, y, column_id="id", column_sort="time", column_value="value",
                                  default_fc_parameters=MinimalFCParameters())
    return X,y

def get_train_test_tsfresh(X,y,steps):
    y2 = pd.DataFrame(y)
    j = 0
    while j < steps:
        y2[f'y_{j}'] = y2['value'].shift(-steps - j)
        j = j + 1

    y2 = y2.dropna(axis=0)
    y2 = y2[y2.index.isin(X.index)]
    X = X[X.index.isin(y2.index)]
    size = int(len(data) * 0.60)
    X_train, X_test = X[0:size], X[size:len(X)]
    y_train, y_test = y2[0:size], y2[size:len(y2)]
    size2 = int(len(X_test) * 0.60)
    X_test_search, X_test_final = X_test[0:size2], X_test[size2:len(X_test)]
    y_test_search, y_test_final = y_test[0:size2], y_test[size2:len(y_test)]
    return X_train,y_train,X_test_search,y_test_search,X_test_final,y_test_final




