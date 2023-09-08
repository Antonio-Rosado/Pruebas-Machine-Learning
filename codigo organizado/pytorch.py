import pandas as pd
#arima, autoarima, decision tree, random forest, adaboost, gaussian process, neural network
#tabla modelo-parámetros-mae-entrada-salida
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import xgboost as xgb
import lightgbm as lgb
import torch
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
from preprocessing import periodic_spline_transformer
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

if __name__ == '__main__':

    class SequenceDataset(Dataset):
        def __init__(self, dataframe, target, features, sequence_length=5):
            self.features = features
            self.target = target
            self.sequence_length = sequence_length
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start:(i + 1), :]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
                x = self.X[0:(i + 1), :]
                x = torch.cat((padding, x), 0)

            return x, self.y[i]


    class ShallowRegressionLSTM(nn.Module):
        def __init__(self, num_sensors, hidden_units):
            super().__init__()
            self.num_sensors = num_sensors  # this is the number of features
            self.hidden_units = hidden_units
            self.num_layers = 1

            self.lstm = nn.LSTM(
                input_size=num_sensors,
                hidden_size=hidden_units,
                batch_first=True,
                num_layers=self.num_layers
            )

            self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

        def forward(self, x):
            batch_size = x.shape[0]
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

            _, (hn, _) = self.lstm(x, (h0, c0))
            out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

            return out


    def train_model(data_loader, model, loss_function, optimizer):
        num_batches = len(data_loader)
        total_loss = 0
        model.train()

        for X, y in data_loader:
            output = model(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}")


    def test_model(data_loader, model, loss_function):

        num_batches = len(data_loader)
        total_loss = 0

        model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                output = model(X)
                total_loss += loss_function(output, y).item()

        avg_loss = total_loss / num_batches
        print(f"Test loss: {avg_loss}")


    def predict(data_loader, model):

        output = torch.tensor([])
        model.eval()
        with torch.no_grad():
            for X, _ in data_loader:
                y_star = model(X)
                output = torch.cat((output, y_star), 0)

        return output


    data = pd.read_excel('datasets/Demanda_2015.xlsx', names=['DATE', 'TIME', 'DEMAND'])
    data['DATE-TIME'] = data.apply(lambda r : pd.datetime.combine(r['DATE'],r['TIME']),1)
    data = data.drop(columns=['DATE','TIME'])
    data = data[['DATE-TIME','DEMAND']]

    data = data.set_index('DATE-TIME')

    data['DEMAND2'] = data ['DEMAND']

    forecast_lead = 30
    target = f"{'DEMAND'}_lead{forecast_lead}"
    features = list(data.columns.difference(['DEMAND']))
    print(features)

    data[target] = data['DEMAND'].shift(-forecast_lead)
    data = data.iloc[:-forecast_lead]

    print(data.describe())
    print(data.head())
    print(data.tail())

    test_start = "2015-10-10 00:00:00"

    df_train = data.loc[:test_start].copy()
    df_test = data.loc[test_start:].copy()

    torch.manual_seed(101)

    batch_size = 4
    sequence_length = 30

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = 5e-5
    num_hidden_units = 16

    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(2):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

    print(df_out)

