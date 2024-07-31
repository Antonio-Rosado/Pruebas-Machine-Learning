from ancillary_functions import save_model,split_train_test_at_point
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import RegressorChain
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from adapt.feature_based import CORAL, PRED, SA, FA, TCA, DANN, fMMD
from adapt.instance_based import TrAdaBoostR2, LDM, NearestNeighborsWeighting, BalancedWeighting
from adapt.parameter_based import RegularTransferNN, RegularTransferLR
from adapt.parameter_based import FineTuning






def adapt_model(X_train1, y_train1,X_train2, y_train2, X_test_final2, y_test_final2, modelname,loaded_model,model_type):

    model = fit_model(modelname,loaded_model,X_train1, y_train1, X_train2,y_train2, model_type)

    predictions = model.predict(X_test_final2)
    if(model_type == 'tf'):
        predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]))

    mse = mean_squared_error(y_test_final2, predictions)
    mae = mean_absolute_error(y_test_final2, predictions)
    print('mse:' + str(mse))
    print('mae:' + str(mae))
    return mse, mae, model

def fit_model(model_name, given_model,X_train1, y_train1, X_train2,y_train2,model_type):
    if (model_name=='coral'):
        model = CORAL(given_model, Xt=X_train2, random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name=='nnw'):
        model = NearestNeighborsWeighting(given_model, Xt=X_train2, n_neighbors=5,random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name=='sa'):
        model = SA(given_model, Xt=X_train2, random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name=='ldm'):
        model = LDM(given_model, Xt=X_train2, random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name=='bw'):
        model = BalancedWeighting(given_model, Xt=X_train2, yt=y_train2,gamma=0.5, random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name=='fa'):
        model = FA(given_model, Xt=X_train2,yt=y_train2, random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name == 'r2'):
        #model = TrAdaBoostR2(given_model,Xt=X_train2, yt=y_train2, n_estimators=10, random_state=0)
        model = LDM(given_model, Xt=X_train2, random_state=0)
        model.fit(X_train1, y_train1)
    if (model_name == 'rt'):
        if(model_type=='tf'):
            model = RegularTransferNN(given_model, loss="mse", lambdas=1., random_state=0)
        if(model_type == 'sk'):
            model = CORAL(given_model, Xt=X_train2, random_state=0)
        model.fit(X_train2, y_train2)
    return model