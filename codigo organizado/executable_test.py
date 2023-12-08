import pandas as pd
from table_creation import obtain_results_tables, obtain_results_tables_tsfresh
from pytorch import get_mse_mae_all


if __name__ == '__main__':

    #Demanda 2015
    data = pd.read_excel('input/Demanda_2015.xlsx', names=['DATE', 'TIME', 'DEMAND'])
    data['DATE-TIME'] = data.apply(lambda r : pd.datetime.combine(r['DATE'],r['TIME']),1)
    data = data.drop(columns=['DATE','TIME'])
    data = data[['DATE-TIME','DEMAND']]

    data = data.set_index('DATE-TIME')

    data['DEMAND2'] = data['DEMAND']

    forecast_lead = 30
    target = f"{'DEMAND'}_lead{forecast_lead}"
    features = list(data.columns.difference(['DEMAND']))
    print(features)

    data[target] = data['DEMAND'].shift(-forecast_lead)
    data = data.iloc[:-forecast_lead]
    test_start = "2015-10-10 00:00:00"

    #Desempleo EEUU
    #data = pd.read_excel('datasets/Desempleo EEUU.xls', names=['DATE', 'UNEMPLOYMENT CLAIMS'])
    #data['DATE'] = pd.to_datetime(data['DATE'])
    #data = data.set_index('DATE')
    #data['UNEMPLOYMENT CLAIMS 2'] = data['UNEMPLOYMENT CLAIMS']

    #forecast_lead = 30
    #target = f"{'UNEMPLOYMENT CLAIMS'}_lead{forecast_lead}"
    #features = list(data.columns.difference(['UNEMPLOYMENT CLAIMS']))
    #print(features)

    #data[target] = data['UNEMPLOYMENT CLAIMS'].shift(-forecast_lead)
    #data = data.iloc[:-forecast_lead]
    #test_start = "2005-01-01"

    # data[target] = data['DEMAND'].shift(-forecast_lead)
    # data = data.iloc[:-forecast_lead]
    # test_start = "2015-10-10 00:00:00"

    #data = pd.read_csv('datasets/Calidad Aire Milan.csv')
    #data['local_datetime'] = pd.to_datetime(data['local_datetime'])
    #data = data.set_index('local_datetime')



    print(data.describe())
    print(data.head())
    print(data.tail())

    config = pd.read_excel('input/configuraciontest.xlsx')
    print(type(config.iloc[0][0]))
    

    #obtain_results_tables(data,config,'DEMAND',fe=False, result_name = "no_fe",hour_week_data = True)
    #obtain_results_tables(data, config, 'pm2p5', fe=True, result_name="fe",hour_week_data = True)
    obtain_results_tables_tsfresh(data, config, 'DEMAND', result_name="tabla_test",lags=15,hour_week_data = True)
    #results = []
    #get_mse_mae_all(data, forecast_lead, target, features, test_start, 'DEMAND', 6, 6, results)


