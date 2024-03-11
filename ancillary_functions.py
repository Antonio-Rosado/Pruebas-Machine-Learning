import pickle



def split_train_test_at_point(data,point, features, target):
    train = data.loc[:point].copy()
    test = data.loc[point:].copy()
    X_train = train[features]
    y_train = train[target]
    X_test_final = test[features]
    y_test_final = test[target]

    return X_train, y_train, X_test_final, y_test_final



def save_model(model, filename):
    pickle.dump(model,open(filename, "wb"))


def load_model(filename):
    return pickle.load(open(filename, "rb"))


def add_lags(df,lags,features):

    for i in range(0, lags+1):
        df['lag_sales_' + str(i)] = df.groupby('id')['Sales'].shift(i)
        features.append('lag_sales_' + str(i))

    df = df.fillna(method='bfill')

    return df


def establish_target(df,outputname,forecast_lead):


    target = f"{outputname}_lead{forecast_lead}"


    df[target] = df.groupby('id')[outputname].shift(-forecast_lead)

    df= df.dropna(subset=[target])

    return df,target