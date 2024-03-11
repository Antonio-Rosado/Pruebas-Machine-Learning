import pandas as pd
import seaborn as sns
import category_encoders as ce
import matplotlib.pyplot as plt



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

clean_df.to_csv("output/" + "modified_m5_testing.csv")


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





