from grid_search import *
from pytorch import SequenceDataset
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
            print(mse)


        mselist.append(mse)

    results.append(mselist)
    return results