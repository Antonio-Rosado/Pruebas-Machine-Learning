import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from ydata_profiling.utils.cache import cache_file
from ydata_profiling.visualisation.plot import timeseries_heatmap
from table_creation import transfer_learning_analysis
from preprocessing import select_data_fragment


sell_prices_df = pd.read_csv('datasets/m5/sell_prices.csv')
train_sales_df = pd.read_csv('datasets/m5/sales_train_validation.csv')
calendar_df = pd.read_csv('datasets/m5/calendar.csv')



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
print(train_sales_cal_df)
train_sales_cal_df_total = train_sales_cal_df
train_sales_cal_df_total['total_sales'] = train_sales_cal_df.sum(axis=1)
train_sales_cal_df_total = train_sales_cal_df_total[['total_sales']]
train_sales_cal_df_total = train_sales_cal_df_total.reset_index()
#train_sales_cal_df_total = train_sales_cal_df_total.set_index('date')
#print(train_sales_cal_df_total)
#print(train_sales_cal_df.iloc[:,30490])


a=30400
b=30405

test_start = "2014-04-04"
test_transfer = "2016-04-04"
filename = 'resultados_m5_fragmento_-17'
transfer_learning_analysis(train_sales_cal_df,a,b,test_start,test_transfer,filename)





