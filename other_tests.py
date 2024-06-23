import pandas as pd
from pytorch_models import create_model_pytorch, skorch_neural_network, pytorch_neural_network, skorch_model
from sklearn_models import create_model_sklearn, skorch_model
from ancillary_functions import add_lags, establish_target, split_train_test_at_point
from adapt.feature_based import CORAL
from tensorflow.python.framework.ops import disable_eager_execution
import time
import random
from tensorflow import keras
from adapt_models import adapt_model
from tensorflow_models import tensorflow_neural_network, test_tensorflow, create_model_tensorflow, compare_models_tensorflow, create_sequences

if __name__ == '__main__':

    df = pd.read_csv("output/" + "modified_m5_testing.csv")
    lags = 5
    features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']

    pd.set_option('display.max_columns', None)

    print(df)

    forecast_lead = 30
    outputname = 'Sales'

    df, target = establish_target(df,outputname,forecast_lead)

    trainingsize = 200

    df = add_lags(df,lags,features)

    print(df)

    test_start = 900
    validation_start = 1400
    train_limit= trainingsize



    data_model = df[df['id'] < train_limit].copy()
    idlist = random.sample(range(0,30489+1), trainingsize)
    data_model2 = df[df['id'].isin(idlist)].copy()
    idList = [range(101,121),range(121,141),range(141,161),range(161,181),range(181,201),range(201,221),range(221,241),range(241,261),range(261,281),range(281,301)]

    df_train = data_model[data_model['d'] < validation_start].copy()
    df_test = data_model[data_model['d'] >= validation_start].copy()
    X_train1, y_train1 = create_sequences(df_train, features, target, 30)
    X_test_final1, y_test_final1 = create_sequences(df_test, features, target, 30)
    #mse, mae, model, mape = tensorflow_neural_network(df_train,df_test,target,features, 4, 30, 'lstm', 10)
    #test_tensorflow(model, df_t, validation_start, 4, 30, target, features, outputname)
    #model = keras.models.load_model('tf_test.keras')
    #test_tensorflow(model, data_model, validation_start, 4, 30, target, features, outputname)
    #skorch_model(data_model, predictions,y_test,validation_start, outputname, features,target, result)
    '''
    ids = idList[0]
    new_df = data_model[data_model['id'].isin(ids)].copy()
    new_df[target] = new_df.groupby('id')[outputname].shift(-forecast_lead)
    new_df = new_df.dropna(subset=[target])
    #mse, mae, model,mape = create_model_tensorflow(data_model, test_start, validation_start, features, 4, 30, 'lstm', 'tf_test.keras',target, 50)
    X_t, y_t = create_sequences(new_df, features, target, 30)
    y_t = y_t.reshape((y_t.shape[0], y_t.shape[1], 1))
    adapt_model = CORAL(model, Xt=X_t, random_state=0)
    adapt_model.fit(X_t, y_t)
    #compare_models_tensorflow(data_model, features, test_start, validation_start, outputname, model, ids, 4, 30, 'lstm', target, forecast_lead, 50)

    #print(result)
    '''
    loaded_model = keras.models.load_model('tf_test.keras')
    adapt_model(X_train1, y_train1, X_train1, y_train1, X_test_final1, y_test_final1, 'lr', loaded_model, 'tf')

