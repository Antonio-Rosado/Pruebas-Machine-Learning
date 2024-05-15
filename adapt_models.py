from ancillary_functions import save_model,split_train_test_at_point
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import RegressorChain
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from adapt.feature_based import CORAL
from adapt.instance_based import TrAdaBoostR2
from adapt.parameter_based import RegularTransferNN






def adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, modelname,loaded_model):

    model = fit_model(modelname,loaded_model,X_train1, y_train1, X_train2,y_train2)

    predictions_final = model.predict(X_test_final2)
    mse = mean_squared_error(y_test_final2, predictions_final)
    mae = mean_absolute_error(y_test_final2, predictions_final)
    print('mse:' + str(mse))
    print('mae:' + str(mae))
    return mse, mae, model

def fit_model(model_name, given_model,X_train1, y_train1, X_train2,y_train2):
    if (model_name=='coral'):
        model = CORAL(given_model, lambda_=1e-3, random_state=0)
        model.fit(X_train1, y_train1, X_train2)
    if (model_name == 'r2'):
        model = TrAdaBoostR2(given_model, n_estimators=30, random_state=0)
        model.fit(X_train1, y_train1, X_train2, y_train2)
    return model