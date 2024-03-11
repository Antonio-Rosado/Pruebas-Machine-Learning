import pandas as pd
import seaborn as sns
import category_encoders as ce
import matplotlib
from sklearn import preprocessing
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from ydata_profiling.utils.cache import cache_file
from ydata_profiling.visualisation.plot import timeseries_heatmap
from table_creation import get_table_pytorch_plus2
from preprocessing import select_data_fragment
from tranfer_learning import create_model_pytorch


sell_prices_df = pd.read_csv('datasets/m5/sell_prices.csv')
train_sales_df = pd.read_csv('datasets/m5/sales_train_validation.csv')
calendar_df = pd.read_csv('datasets/m5/calendar.csv')

pd.set_option('display.max_columns', None)

print(sell_prices_df)
print(train_sales_df)
print(calendar_df)

value_columns = train_sales_df.iloc[:,6:].columns.tolist()

print(value_columns)


encoder1 = ce.BinaryEncoder(cols=['item_id','dept_id','cat_id','store_id', 'state_id'])
encoder2 = ce.BinaryEncoder(cols=['event_name_1','event_name_2', 'event_type_1','event_type_2'])

sales_encoded = encoder1.fit_transform(train_sales_df)

sales_encoded['id'] = sales_encoded.index

print(sales_encoded)

feature_columns = [item for item in sales_encoded.columns.tolist() if item not in value_columns]

print(feature_columns)

sales_melt = sales_encoded.melt(id_vars=feature_columns, var_name='Day', value_name='Sales')

print(sales_melt)

calendar_df.date = pd.to_datetime(calendar_df.date)

calendar_df['mday'] = [calendar_df['date'][i].day for i in range(len(calendar_df))]

calendar_encoded = encoder2.fit_transform(calendar_df)

sales_calendar = sales_melt.merge(calendar_encoded, left_on='Day', right_on='d')

clean_df = sales_calendar.drop(['date','Day','weekday'], axis=1)

clean_df['d'] = clean_df['d'].str.extract('(\d+)',expand=False).astype(int)

#clean_df.to_csv("output/" + "modified_m5.csv")


print(clean_df.iloc[0:50,:])

for col in ['weekday','month','year','event_name_1','event_name_2','event_type_1','event_type_2']:
    sns.countplot(calendar_df, x=col)
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()

for col in ['store_id','state_id','cat_id','dept_id']:
    sns.countplot(train_sales_df, x=col)
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()



d_cols = [c for c in train_sales_df.columns if 'd_' in c]
train_sales_df['total_sales_all_days'] = train_sales_df[d_cols].sum(axis = 1)
train_sales_df['avg_sales_all_days'] = train_sales_df[d_cols].mean(axis = 1)
train_sales_df['median_sales_all_days'] = train_sales_df[d_cols].median(axis = 1)
train_sales_df.groupby(['id'])['total_sales_all_days'].sum().sort_values(ascending=False)
df_agg = pd.DataFrame(train_sales_df.groupby(['id', 'cat_id', 'store_id'])['total_sales_all_days'].sum().sort_values(ascending=False))
sell_prices_df['category'] = sell_prices_df['item_id'].str.split("_", expand=True)[0]
train_sales_prices_df = train_sales_df.merge(sell_prices_df, how='inner', left_index=True, right_index=True, validate="1:1")
calendar_df.date = pd.to_datetime(calendar_df.date)
train_sales_cal_df = train_sales_df.set_index('id')[d_cols].T.merge(calendar_df.set_index('d')['date'], left_index=True, right_index=True,validate="1:1").set_index('date')
train_sales_cal_df_total = train_sales_cal_df
train_sales_cal_df_total['total_sales'] = train_sales_cal_df.sum(axis=1)
train_sales_cal_df_total = train_sales_cal_df_total[['total_sales']]
train_sales_cal_df_total = train_sales_cal_df_total.reset_index()
#train_sales_cal_df_total = train_sales_cal_df_total.set_index('date')
#print(train_sales_cal_df_total)
#print(train_sales_cal_df.iloc[:,30490])



test_start = 900
validation_start = 1400
train_limit= 100
features = ['d', 'mday', 'month', 'wday', 'wm_yr_wk', 'year']
model_file = 'models/lstm_one_item.pickle'
data_model = clean_df[clean_df['id'] < train_limit].copy()
#create_model_pytorch(data_model,test_start,validation_start,'Sales', features, 4, 30, 'lstm', model_file)





