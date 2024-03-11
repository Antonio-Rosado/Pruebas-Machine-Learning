from grid_search import *
from pytorch import get_mse_mae_all, pytorch_neural_network, mse_basic, get_mse_mae_all, save_model, load_model
from preprocessing import feature_engineering
from tranfer_learning import apply_tranfer_learning, test_pytorch, test_sklearn


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

def get_table_pytorch_plus (dataset, test_start,test_transfer, filename):

    results = []
    results_tf = []
    best_models = {}

    df_train = dataset[dataset['Day'] < test_start].copy()
    df_test = dataset[dataset['Day'] >= test_start].copy()
    df_transfer = dataset[dataset['Day'] >= test_transfer].copy()

    for i in range(0, len(dataset.columns)):
        data = dataset.iloc[:, i]
        outputname = data.name
        data = data.to_frame()
        print(data.to_string())
        print(data.index)
        data[outputname + '2'] = data[outputname]


        forecast_lead = 30
        target = f"{outputname}_lead{forecast_lead}"
        features = list(data.columns.difference([outputname]))
        print(features)

        data[target] = data[outputname].shift(-forecast_lead)
        data = data.iloc[:-forecast_lead]

        results,best_models = get_mse_mae_all(data, target, features, test_start, outputname, results,best_models)

    print(best_models)

    for i in range(0, len(df_transfer.columns)):
        data = df_transfer.iloc[:, i]
        outputname = data.name
        data = data.to_frame()
        feature_engineering(data, outputname)
        data[outputname + '2'] = data[outputname]
        forecast_lead = 30
        target = f"{outputname}_lead{forecast_lead}"
        features = list(data.columns.difference([outputname]))
        data[target] = data[outputname].shift(-forecast_lead)
        data = data.iloc[:-forecast_lead]
        results_tf = apply_tranfer_learning(best_models, data, 6, 6, target, features,outputname, results_tf)

    columns = ['time_series', 'CNN_mse', 'CNN_mae', 'LSTM_mse','LSTM_mae','Transformer_mse','Transformer_mae','XGB_mse','XGB_mae','RandomForest_mse','RandomForest_mae']
    df1 = pd.DataFrame(results, columns=columns)
    df1.set_index(['time_series'], inplace=True)
    print(df1)

    df1.to_excel("output/" + filename + ".xlsx")

    columns_tf = ['series']
    for k in best_models.keys():
        columns_tf.append(k)
    print(columns_tf)
    print(results_tf)
    df2 = pd.DataFrame(results_tf, columns=columns_tf)
    df2.set_index(['series'], inplace=True)
    print(df2)

    df2.to_excel("output/" + filename + "transfer_learning.xlsx")





def transfer_learning_analysis(dataframe,data_start,data_end,test_start,test_transfer,filename):

    df = select_data_fragment(dataframe, data_start, data_end)
    for col in df.columns:
        df[col].plot()
        plt.title(col)
        plt.show()
    print(df)

    get_table_pytorch_plus(df, test_start, test_transfer, filename)


def get_table_pytorch_plus2 (dataset, test_start,validation_start,train_limit, outputname):

    data_model = dataset[dataset['id'] < train_limit].copy()
    df_transfer = dataset[dataset['id'] >= train_limit].copy()

    forecast_lead = 30
    target = f"{outputname}_lead{forecast_lead}"


    features = ['d','mday', 'month', 'wday', 'wm_yr_wk', 'year']
    print(features)

    data_model[target] = data_model.groupby('id')[outputname].shift(-forecast_lead)

    print(data_model['Sales_lead30'])
    data_model = data_model.dropna(subset=['Sales_lead30'])
    print(data_model['Sales_lead30'])
    print(data_model)
    mse, mae, model = pytorch_neural_network(data_model, target, features, validation_start,test_start, outputname, 4, 30, 'lstm')
    #mse, mae, model = mse_basic(data_model, features, target, 'forest',test_start)
    #get_mse_mae_all(data_model, target, features, test_start, outputname, results, best_models)

    model_file = 'modelexample.pickle'

    #save_model(model,model_file)

    loaded_model = load_model(model_file)

    for i in range(101,105):

        new_df = df_transfer[df_transfer['id'] == 150].copy()

        new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)

        new_df = new_df.dropna(subset=['Sales_lead30'])

        mse1, mae, model = pytorch_neural_network(new_df, target, features, test_start,validation_start, outputname, 4, 30, 'lstm')

        mse2 = test_pytorch(loaded_model, new_df,validation_start,4,30, target, features,outputname)

        print('MSE without tansfer:' + str(mse1))

        print('MSE with tansfer:' + str(mse2))


    #print(mse)
    #print(mae)
    #print(model)
    #print(loaded_model)
