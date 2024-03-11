from ancillary_functions import save_model,split_train_test_at_point
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import RegressorChain
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor






def test_sklearn(model, data,test_start, target, features,outputname):

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


def create_model_sklearn(data_model, test_start,validation_start, outputname, features,modelname, model_file, target):


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


def compare_models_sklearn(df_transfer,features,test_start,validation_start, outputname,loaded_model, productIds, modelname, target,forecast_lead):

    new_df = df_transfer[df_transfer['id'].isin(productIds)].copy()

    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)

    new_df = new_df.dropna(subset=[target])

    X_train, y_train, X_test_final, y_test_final = split_train_test_at_point(new_df, test_start, features, target)

    mse1, mae1, model,mape = sklearn_model(new_df, X_train, y_train, X_test_final, y_test_final,validation_start,'Sales', features,target,modelname)

    mse2, mae2, mape2 = test_sklearn(loaded_model, new_df,validation_start, target, features,'Sales')

    print('MSE without tansfer:' + str(mse1))

    print('MSE with tansfer:' + str(mse2))

    return mse1, mse2, mae1, mae2

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