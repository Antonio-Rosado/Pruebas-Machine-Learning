import pandas as pd
#arima, autoarima, decision tree, random forest, adaboost, gaussian process, neural network
#tabla modelo-parámetros-mae-entrada-salida
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import RegressorChain
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import DecomposeResult
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn import tree
from tbats import TBATS, BATS
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
data = pd.read_excel('Demanda_2015.xlsx', names=['DATE', 'TIME', 'DEMAND'])
data['DATE-TIME'] = data.apply(lambda r : pd.datetime.combine(r['DATE'],r['TIME']),1)
data = data.drop(columns=['DATE','TIME'])
data = data[['DATE-TIME','DEMAND']]
data = data.set_index('DATE-TIME')
'''
data['TIME-STEP'] = np.arange(len(data.index))
data['LAG'] = data['DEMAND'].shift(1)
'''
print(data.describe())
print(data.head())
print(data.tail())


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




'''
plt.figure(figsize=(16,6))
sns.lineplot(x='TIME-STEP', y='DEMAND',data=data.iloc[0:1008])
plt.show()
sns.lineplot(x='TIME-STEP', y='DEMAND',data=data.iloc[0:8496])
plt.show()
sns.lineplot(x='TIME-STEP', y='DEMAND', data=data)
plt.show()
pd.plotting.autocorrelation_plot(data)
plt.show()
'''

'''
train = train.drop(columns=['TIME-STEP','LAG'])
test = test.drop(columns=['TIME-STEP','LAG'])
model = ARIMA(train, order=(6,6,6))
model_fit = model.fit()
predictions = model_fit.forecast(steps=17871)
'''

'''
train = train.drop(columns=['TIME-STEP','LAG'])
test = test.drop(columns=['TIME-STEP','LAG'])
model = auto_arima(train, start_p=0, start_q=0, m=6, start_P=0, seasonal=True,
                        trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
model.fit(train)
predictions = model.predict(n_periods=17871)
'''
#data: pandas dataframe that contains the dataset
#model: scikit-learn model to use, with fit and predict methods
#training_percentage: percentage of the dataset to be used for training, the rest will be used for testing
#output:name of the output
#lags:number of lags to use as input
#steps:number of steps to predict as output
def train_test_mae(data,model,training_percentage,outputname, lags, steps):
    formatted_data = window_input_output(lags, steps, data, outputname)
    X_train, y_train, X_test, y_test = split_train_test(formatted_data, outputname,training_percentage)
    mae = get_mae(X_train, y_train, X_test, y_test, model)
    return mae


def split_train_test(data,outputname, training_percentage):
    size = int(len(data) * training_percentage)
    train, test = data[0:size], data[size:len(data)]
    X_cols = [col for col in data.columns if col.startswith('x')]
    X_cols.insert(0, outputname)
    y_cols = [col for col in data.columns if col.startswith('y')]
    X_train = train[X_cols].values
    y_train = train[y_cols].values
    X_test = test[X_cols].values
    y_test = test[y_cols].values
    return X_train, y_train, X_test, y_test



def obtain_results_table(data, modelarrays, outputname):
    results = []
    for modelarray in modelarrays:
        model = modelarray[0]
        model.random_state = 11
        trainingpercent = modelarray[1]
        output = outputname
        lags = modelarray[2]
        steps = modelarray[3]
        model_params = modelarray[4]
        mae = train_test_mae(data, model, trainingpercent, output, lags, steps)
        results.append([type(model).__name__,model_params,trainingpercent,lags, steps, mae])
        df = pd.DataFrame(results, columns=['model', 'parameters', 'training_percentage', 'lags_used','steps_forecasted', 'mae'])
    print(df)
    df.to_excel("results.xlsx")


def get_mae(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(model.predict(X_test))
    print(y_test)
    print(type(model).__name__)
    print('Test MAE: %.3f' % mae)
    return mae

modelarrays = []
tree = tree.DecisionTreeRegressor()
forest = RandomForestRegressor(n_estimators=100)
adaboost = AdaBoostRegressor(n_estimators=50)
adaboostseq = RegressorChain(adaboost)
mlp = MLPRegressor( max_iter=500)
#model = GaussianProcessRegressor(kernel=RBF(),random_state=0)

modelarrays.append([tree, 0.66, 5, 5, ''])
modelarrays.append([tree, 0.66, 10, 5,''])
modelarrays.append([tree, 0.66, 5, 7,''])
modelarrays.append([tree, 0.55, 5, 5,''])
modelarrays.append([mlp, 0.66, 5, 5,'max_iter=500'])
modelarrays.append([mlp, 0.66, 10, 5,'max_iter=500'])
modelarrays.append([mlp, 0.66, 5, 7,'max_iter=500'])
modelarrays.append([mlp, 0.55, 5, 5,'max_iter=500'])
modelarrays.append([forest, 0.66, 5, 5,'n_estimators=100'])
modelarrays.append([forest, 0.66, 10, 5,'n_estimators=100'])
modelarrays.append([forest, 0.66, 5, 7,'n_estimators=100'])
modelarrays.append([forest, 0.55, 5, 5,'n_estimators=100'])
modelarrays.append([adaboostseq, 0.66, 5, 5,'n_estimators=50'])
modelarrays.append([adaboostseq, 0.66, 10, 5,'n_estimators=50'])
modelarrays.append([adaboostseq, 0.66, 5, 7,'n_estimators=50'])
modelarrays.append([adaboostseq, 0.55, 5, 5,'n_estimators=50'])


obtain_results_table(data,modelarrays,'DEMAND')





