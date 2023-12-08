from grid_search import *
from pytorch import get_mse_mae_all


def obtain_results_tables(data, config, outputname,fe, result_name,hour_week_data):
    results1 = []
    data = data[:-12]
    for i in range(0,len(config)):
        model,model_params,maxlags,steps = build_model(config.iloc[i])
        model.random_state = 11
        output = outputname
        results1= fill_results_table( model, model_params, data, outputname, maxlags, steps, fe, results1,hour_week_data)

    columns1 = ['model', 'lags_used', 'parameters', 'steps_forecasted','mae','mse','mape']
    if (hour_week_data):
        for i in range(0, 7):
            columns1.append("mae_day_" + str(i))
        for i in range(0,24):
            columns1.append("mae_hour_"+str(i))
    df1 = pd.DataFrame(results1, columns=columns1)
    df1.set_index(['model','steps_forecasted','lags_used','parameters'], inplace=True)
    print(df1)
    df1.to_excel("output/" +result_name + "_resultados.xlsx")


def obtain_results_tables_tsfresh(data, config, outputname, result_name,lags,hour_week_data):

    X,y = extract_feautres_tsfresh(data, outputname,lags)
    size = int(len(data) * 0.60)
    results1 = []
    for i in range(0, len(config)):
        model, model_params, maxlags, steps = build_model(config.iloc[i])
        X_train, y_train, X_test_search, y_test_search, X_test_final, y_test_final = get_train_test_tsfresh(X, y, steps,size)

        print(model_params)
        model.random_state = 11
        output = outputname
        results1 = grid_search_full(model,X_train,y_train,X_test_final,y_test_final,model_params,results1,lags,steps,hour_week_data)

    columns1 = ['model', 'lags_used', 'parameters', 'steps_forecasted','mae','mse','mape']
    if (hour_week_data):
        for i in range(0,7):
            columns1.append("mae_day_"+str(i))
        for i in range(0,24):
            columns1.append("mae_hour_"+str(i))
    df1 = pd.DataFrame(results1, columns=columns1)
    df1.set_index(['model', 'steps_forecasted', 'lags_used', 'parameters'], inplace=True)
    print(df1)
    df1.to_excel("output/" + result_name + "_resultados_tsfresh.xlsx")

def get_table_pytorch_plus (dataset, test_start, filename):

    results = []
    for i in range(0, len(dataset.columns)):
        data = dataset.iloc[:, i]
        outputname = data.name
        data = data.to_frame()
        print(data.to_string())
        data[outputname + '2'] = data[outputname]

        forecast_lead = 30
        target = f"{outputname}_lead{forecast_lead}"
        features = list(data.columns.difference([outputname]))
        print(features)

        data[target] = data[outputname].shift(-forecast_lead)
        data = data.iloc[:-forecast_lead]
        test_start = "2015-04-04"

        results = get_mse_mae_all(data, forecast_lead, target, features, test_start, outputname, 6, 6, results)

    columns = ['time_series', 'CNN_mse', 'CNN_mae', 'LSTM_mse','LSTM_mae','Transformer_mse','Transformer_mae','XGB_mse','XGB_mae','RandomForest_mse','RandomForest_mae']
    df1 = pd.DataFrame(results, columns=columns)
    df1.set_index(['time_series'], inplace=True)
    print(df1)
    df1.to_excel("output/" + filename + ".xlsx")





