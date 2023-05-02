import pandas as pd
#arima, autoarima, decision tree, random forest, adaboost, gaussian process, neural network
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
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
data['TIME-STEP'] = np.arange(len(data.index))
data['LAG'] = data['DEMAND'].shift(1)
print(data.describe())
print(data.head())
print(data.tail())
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


size = int(len(data) * 0.66)
train, test = data[0:size], data[size:len(data)]

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



X_train = train.loc[:, ['LAG']]  # features
y_train = train.loc[:, 'DEMAND']  # target
X_test = test.loc[:, ['LAG']]  # features
y_test = test.loc[:, 'DEMAND']  # target
X_train['LAG'][0]=X_train['LAG'][1]
print(X_train)
print(y_train)
print(X_test)
print(y_test)
model = tree.DecisionTreeRegressor()
model.fit(X_train, y_train)
print(model.predict(X_test))
predictions = pd.Series(model.predict(X_test), index=X_test.index)
test=y_test


'''
X_train = train.loc[:, ['LAG']]  # features
y_train = train.loc[:, 'DEMAND']  # target
X_test = test.loc[:, ['LAG']]  # features
y_test = test.loc[:, 'DEMAND']  # target
X_train['LAG'][0]=X_train['LAG'][1]
model = RandomForestRegressor(n_estimators=3)
model.fit(X_train, y_train)
predictions = pd.Series(model.predict(X_test), index=X_test.index)
test=y_test
'''

'''
X_train = train.loc[:, ['LAG']]  # features
y_train = train.loc[:, 'DEMAND']  # target
X_test = test.loc[:, ['LAG']]  # features
y_test = test.loc[:, 'DEMAND']  # target
X_train['LAG'][0]=X_train['LAG'][1]
model = AdaBoostRegressor(n_estimators=3)
model.fit(X_train, y_train)
predictions = pd.Series(model.predict(X_test), index=X_test.index)
test=y_test
'''

'''
X_train = train.loc[:, ['LAG']]  # features
y_train = train.loc[:, 'DEMAND']  # target
X_test = test.loc[:, ['LAG']]  # features
y_test = test.loc[:, 'DEMAND']  # target
X_train['LAG'][0]=X_train['LAG'][1]
kernel = RBF()
model = GaussianProcessRegressor(kernel=kernel,
        random_state=0)
model.fit(X_train, y_train)
predictions = pd.Series(model.predict(X_test), index=X_test.index)
test=y_test
'''

'''
X_train = train.loc[:, ['LAG']]  # features
y_train = train.loc[:, 'DEMAND']  # target
X_test = test.loc[:, ['LAG']]  # features
y_test = test.loc[:, 'DEMAND']  # target
X_train['LAG'][0]=X_train['LAG'][1]
model = MLPRegressor(random_state=1, max_iter=500)
model.fit(X_train, y_train)
predictions = pd.Series(model.predict(X_test), index=X_test.index)
test=y_test
'''

mae = mean_absolute_error(test, predictions)
print('Test MAE: %.3f' % mae)
ax = test.plot()
ax = predictions.plot(ax=ax, linewidth=3)
plt.show()