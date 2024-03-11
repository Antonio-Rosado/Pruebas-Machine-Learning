import math
from ancillary_functions import save_model
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error





class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        print(dataframe[features])
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
        self.num_layers = 3

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


class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet, self).__init__()
        self.conv1d = nn.Conv1d(3, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(704, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.linear2 = nn.Linear(6,1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output).flatten()
        output = self.linear2(output)
        return output

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



def pytorch_neural_network(data,target,features,test_start,validation_start, outputname, batch_size, sequence_length, n_type):
    df_train = data[data['d'] < test_start].copy()
    df_t = data[data['d'] >= test_start].copy()
    df_test = df_t[df_t['d'] < validation_start].copy()
    df_val = df_t[df_t['d'] >= validation_start].copy()
    print(df_train)
    print(df_test)
    print(df_val)

    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev
        df_val[c] = (df_val[c] - mean) / stdev

    torch.manual_seed(101)

    batch_size = batch_size
    sequence_length = sequence_length

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

    val_dataset = SequenceDataset(
        df_val,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(test_loader)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = 5e-5


    model = select_network(n_type, features)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(50):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_val[ystar_col] = predict(val_loader, model).numpy()



    df_out = pd.concat((df_train, df_val))[[target, ystar_col]]


    df_out[outputname + '_lead30'] = df_out[outputname + '_lead30'] * target_stdev + target_mean

    print(df_out)

    mse = mean_squared_error(df_out[outputname + '_lead30'], df_out['Model forecast'])
    print(mse)
    mae = mean_absolute_error(df_out[outputname + '_lead30'], df_out['Model forecast'])
    print(mae)
    mape = mean_absolute_percentage_error(df_out[outputname + '_lead30'], df_out['Model forecast'])
    print(mape)

    return mse, mae, model, mape

def select_network(nn_type, features):
    if (nn_type == 'transformer'):
        ntokens = 1
        emsize = 200
        d_hid = 200
        nlayers = 2
        nhead = 2
        dropout = 0.2
        model = Transformer(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    if (nn_type == 'lstm'):
        num_hidden_units = 32
        model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    if(nn_type == 'cnn'):
        model = CNN_ForecastNet()
    return model


def test_pytorch(model, data,test_start, batch_size,sequence_length, target, features,outputname):


    df_test = data[data['d'] >= test_start].copy()
    df = df_test.copy()

    target_mean = df[target].mean()
    target_stdev = df[target].std()

    for c in df.columns:
        mean = df[c].mean()
        stdev = df[c].std()

        df[c] = (df[c] - mean) / stdev

    test_dataset = SequenceDataset(
        df,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    ystar_col = "Model forecast"
    output = torch.tensor([])
    model.eval()

    with torch.no_grad():

        for X, _ in test_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
        df[ystar_col] = output

    print('setp 4.1')
    df[outputname + '_lead30'] = df[outputname + '_lead30'] * target_stdev + target_mean


    print('setp 4')

    mse = mean_squared_error(df[outputname+ '_lead30'], df['Model forecast'])
    print(mse)
    mae = mean_absolute_error(df[outputname+ '_lead30'], df['Model forecast'])
    print(mae)
    mape = mean_absolute_percentage_error(df[outputname+ '_lead30'], df['Model forecast'])
    print(mape)

    return mse,mae,mape



def create_model_pytorch(data_model, test_start,validation_start, outputname, features,batch_size,sequence_lenght, nn_type, model_file, target):


    mse, mae, model,mape = pytorch_neural_network(data_model, target, features, test_start, validation_start, outputname, batch_size,
                                             sequence_lenght, nn_type)

    save_model(model,model_file)

    return mse, mae, model,mape


def compare_models_pytorch(df_transfer,features,test_start,validation_start, outputname,loaded_model, productIds,batch_size, sequence_lenght, nntype, target, forecast_lead):


    new_df = df_transfer[df_transfer['id'].isin(productIds)].copy()

    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)

    new_df = new_df.dropna(subset=[target])


    pd.set_option('display.max_columns', None)
    print(new_df)

    mse1, mae1, model,mape = pytorch_neural_network(new_df, target, features, test_start, validation_start, outputname, batch_size, sequence_lenght,
                                              nntype)

    mse2, mae2, mape2 = test_pytorch(loaded_model, new_df, validation_start, batch_size, sequence_lenght, target, features, outputname)

    print('MSE without tansfer:' + str(mse1))

    print('MSE with tansfer:' + str(mse2))

    return mse1, mse2, mae1, mae2



