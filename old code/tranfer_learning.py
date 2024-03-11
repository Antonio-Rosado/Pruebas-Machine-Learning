from grid_search import *
from pytorch import SequenceDataset, pytorch_neural_network, save_model
import torch as torch
from torch.utils.data import DataLoader




def apply_tranfer_learning(models, df_test, batch_size,sequence_length, target, features,outputname, results):

    series = models.keys()

    mselist = [outputname]

    print(df_test)
    for s in series:

        original_model_name = s
        model = models[s][0]
        old_mse = models[outputname][1]
        print(model)
        if (model.__module__=='pytorch'):


            df = df_test
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

            df = df * target_stdev + target_mean

            mse = mean_squared_error(df[outputname], df['Model forecast'])
            print('MSE without tansfer:' + str(old_mse))
            print('MSE with tansfer:' + str(mse))

        else:
            df = df_test
            print(outputname)
            print(original_model_name)
            X_test = df[features]
            y_test = df[target]
            print(X_test)
            print(y_test)
            X_test = X_test.rename(columns={outputname +'2': original_model_name+ '2'})

            print(X_test)
            print(y_test)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print('MSE without tansfer:' + str(old_mse))
            print('MSE with tansfer:' + str(mse))


        mselist.append(mse)

    results.append(mselist)
    return results


def test_pytorch(model, data,test_start, batch_size,sequence_length, target, features,outputname):

    df_train = data[data['d'] < test_start].copy()
    df_test = data[data['d'] >= test_start].copy()
    df = df_test.copy()
    print('setp 0')
    target_mean = df[target].mean()
    target_stdev = df[target].std()
    print('setp 1')
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

    print('setp 2')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('setp 3')

    ystar_col = "Model forecast"
    output = torch.tensor([])
    model.eval()

    print('setp 4.0')
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


def test_sklearn(model, data,test_start, target, features,outputname):

    df_train = data[data['d'] < test_start].copy()
    df_test = data[data['d'] >= test_start].copy()

    df = df_test
    print(outputname)
    X_test = df[features]
    y_test = df[target]

    print(X_test)
    print(y_test)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test,predictions)
    mape = mean_absolute_percentage_error(y_test,predictions)
    return mse,mae, mape

def create_model_pytorch(data_model, test_start,validation_start, outputname, features,batch_size,sequence_lenght, nn_type, model_file):

    forecast_lead = 30
    target = f"{outputname}_lead{forecast_lead}"


    print(features)

    data_model[target] = data_model.groupby('id')[outputname].shift(-forecast_lead)

    print(data_model['Sales_lead30'])
    data_model = data_model.dropna(subset=['Sales_lead30'])
    print(data_model['Sales_lead30'])
    print(data_model)
    mse, mae, model,mape = pytorch_neural_network(data_model, target, features, test_start, validation_start, outputname, batch_size,
                                             sequence_lenght, nn_type)

    save_model(model,model_file)

    return mse, mae, model,mape


def create_model_sklearn(data_model, test_start,validation_start, outputname, features,modelname, model_file):

    forecast_lead = 30
    target = f"{outputname}_lead{forecast_lead}"

    print(features)

    data_model[target] = data_model.groupby('id')[outputname].shift(-forecast_lead)

    print(data_model['Sales_lead30'])
    data_model = data_model.dropna(subset=['Sales_lead30'])
    print(data_model['Sales_lead30'])
    print(data_model)

    X_train, y_train, X_test_final, y_test_final = split_train_test_at_point(data_model, test_start, features, target)
    mse, mae, model, mape = sklearn_model(data_model, X_train, y_train, X_test_final,y_test_final,validation_start, outputname, features,target, modelname)
    print(mse)
    print(mae)
    mse, mae, mape = test_sklearn(model,data_model,validation_start, target, features,outputname)
    save_model(model, model_file)
    return mse, mae, model, mape


def sklearn_model(data_model, X_train, y_train, X_test_final,y_test_final,validation_start, outputname, features,target, modelname):

    model = select_model(modelname)
    model.fit(X_train, y_train)
    predictions_final = model.predict(X_test_final)
    mse = mean_squared_error(y_test_final, predictions_final)
    mae = mean_absolute_error(y_test_final, predictions_final)
    print(mse)
    print(mae)
    mse, mae, mape = test_sklearn(model,data_model,validation_start, target, features,outputname)
    return mse, mae, model, mape


def compare_models_pytorch(df_transfer,features,test_start,validation_start, outputname,loaded_model, productIds,batch_size, sequence_lenght, nntype):

    forecast_lead = 30
    target = f"{outputname}_lead{forecast_lead}"

    print(features)

    df_transfer[target] = df_transfer.groupby('id')[outputname].shift(-forecast_lead)

    df_transfer = df_transfer.dropna(subset=['Sales_lead30'])

    new_df = df_transfer[df_transfer['id'].isin(productIds)].copy()

    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)

    new_df = new_df.dropna(subset=['Sales_lead30'])

    print('this')
    pd.set_option('display.max_columns', None)
    print(new_df)

    mse1, mae1, model,mape = pytorch_neural_network(new_df, target, features, test_start, validation_start, outputname, batch_size, sequence_lenght,
                                              nntype)

    mse2, mae2, mape2 = test_pytorch(loaded_model, new_df, validation_start, batch_size, sequence_lenght, target, features, outputname)

    print('MSE without tansfer:' + str(mse1))

    print('MSE with tansfer:' + str(mse2))

    return mse1, mse2, mae1, mae2


def compare_models_sklearn(df_transfer,features,test_start,validation_start, outputname,loaded_model, productId, modelname):

    forecast_lead = 30
    target = f"{outputname}_lead{forecast_lead}"

    print(features)

    df_transfer[target] = df_transfer.groupby('id')[outputname].shift(-forecast_lead)

    df_transfer = df_transfer.dropna(subset=['Sales_lead30'])

    new_df = df_transfer[df_transfer['id'] == productId].copy()

    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)

    new_df = new_df.dropna(subset=['Sales_lead30'])

    X_train, y_train, X_test_final, y_test_final = split_train_test_at_point(new_df, test_start, features, target)

    mse1, mae1, model,mape = sklearn_model(new_df, X_train, y_train, X_test_final, y_test_final,validation_start,'Sales', features,target,modelname)

    mse2, mae2, mape2 = test_sklearn(loaded_model, new_df,validation_start, target, features,'Sales')

    print('MSE without tansfer:' + str(mse1))

    print('MSE with tansfer:' + str(mse2))

    return mse1, mse2, mae1, mae2