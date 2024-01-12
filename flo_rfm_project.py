##############################################################################################
#                          DATA UNDERSTANDING AND PREPARATION                                #
##############################################################################################
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 500)
# pd.set_option('display_max_rows',None)
pd.set_option('display.float_format',lambda x : '%.5f' % x)


df_= pd.read_csv("flo_rfm_project/flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head(10)
df.columns
df.interested_in_categories_12.value_counts()
df.order_channel.value_counts()
df.nunique()
df.describe().T
df.isnull().sum()
df.info()

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df


def check_df(df, head=5):
    print("--------------------- Shape ---------------------")
    print(df.shape)
    print("--------------------- Types ---------------------")
    print(df.dtypes)
    print("--------------------- Head ---------------------")
    print(df.head(head))
    print("--------------------- Missing Values Analysis ---------------------")
    print(missing_values_analysis(df))
    print("--------------------- Quantiles ---------------------")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df, head=10)

df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

# Total purchases for omnichannel customers
df["order_num_total_online_offline"] = df["order_num_total_ever_online"] +  df["order_num_total_ever_offline"]

# Total expense for omnichannel customers
df["customer_onoff_total_expense"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# Review the data type for variable expressing the date
df.info()

dates = ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[dates] = df[dates].apply(pd.to_datetime)

df["last_order_date"].max()


df["last_order_date"].max()
# Out[90]: '2021-05-30'
last_date = dt.datetime(2021,5,30)
type(last_date)
# Out[98]: datetime.datetime

df["first_order_date"]
today_date = dt.datetime(2021, 6, 2)

df.info()

# Distribution of the number of Customers in the Shopping Channels,
# the total number of products purchased and their total expenditures
df.groupby('order_channel').agg({'order_num_total_online_offline': 'sum',
                                'customer_onoff_total_expense':'count'}).sort_values(by="customer_onoff_total_expense",ascending=False)


df.groupby('order_channel').agg({'order_num_total_online_offline': ['count','sum'],
                                'customer_onoff_total_expense':['count','sum']})
df["master_id"].nunique()

df.groupby('last_order_channel').agg({'order_num_total_online_offline': 'sum',
                                'customer_onoff_total_expense':'count'}).sort_values(by="customer_onoff_total_expense",ascending=False)


# Top 10 Customers with the Most Profits
df.groupby('master_id').agg({'customer_onoff_total_expense':'sum'}).\
    sort_values(by="customer_onoff_total_expense",ascending=False).head(10)

# Top 10 Customers with Most Orders
df.groupby('master_id').agg({'order_num_total_online_offline':'sum'}).\
    sort_values(by="order_num_total_online_offline",ascending=False).head(10)

df.groupby('master_id').agg({'order_num_total_online_offline': 'sum',
                                'customer_onoff_total_expense':'sum'}).sort_values(by="customer_onoff_total_expense",ascending=False).head(10)

# Data Preparation Process Function
def data_preparation(dataframe):
    dataframe["order_num_total_online_offline"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]

    dataframe["customer_onoff_total_expense"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    dataframe[dates] = dataframe[dates].apply(pd.to_datetime)

    return dataframe

data_preparation(df)

df.groupby("master_id")["customer_onoff_total_expense"].sum().sort_values(ascending=False).head()

total_categories_expense = df.groupby("interested_in_categories_12")["customer_onoff_total_expense"].sum().sort_values(
    ascending=False).reset_index().head()

total_categories_expense.head()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.barplot(data=total_categories_expense, x='interested_in_categories_12', y='customer_onoff_total_expense')
plt.show(block=True)

##############################################################################################
#                          CALCULATING RFM METRICS                                           #
##############################################################################################
df["last_order_date"].max()
# Out[90]: '2021-05-30'
last_date = dt.datetime(2021,5,30)
type(last_date)
today_date = dt.datetime(2021, 6, 2)
rfm = df.groupby('master_id').agg( {'last_order_date': lambda last_order_date : (today_date- last_order_date.max()).days,
                                    'order_num_total_online_offline': lambda total_purchases : total_purchases.sum(),
                                    'customer_onoff_total_expense': lambda total_expense: total_expense.sum(),
                                    })

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.head(10)

##############################################################################################
#                           CALCULATING RFM SCORES                                           #
##############################################################################################
rfm.describe()

rfm['Recency_score'] = pd.qcut(rfm['Recency'],5, labels = [5,4,3,2,1] )
rfm["Frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()
rfm['RFM_score'] = (rfm['Recency_score'].astype(str)+ rfm['Frequency_score'].astype(str))
rfm[rfm["RFM_score"] == "55"].head(10)
##############################################################################################
#                               DEFINING RFM SCORE AS A SEGMENT                              #
##############################################################################################
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
seg_map
rfm.head()
rfm['segment'] = rfm['RFM_score'].replace(seg_map, regex=True)
rfm[["segment", "RFM_score"]].groupby("segment").agg(["mean", "count"])
rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])
rfm[rfm["segment"] == "need_attention"].head(10)
rfm.head()
rfm.info()
## Write a csv file to a new folder
from pathlib import Path
filepath = Path('D:/12thTerm_DS_Bootcamp/3Week_CRM_Analytics/rfm.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
rfm.to_csv(filepath)


##############################################################################################
#                                     Example Case                                           #
##############################################################################################
#  A new women's shoe brand will be included. The target audience (champions,
# loyal_customers) and women are determined as shoppers.
# We need access to the id numbers of these customers.

vip_cust = (rfm[(rfm["segment"]=="champions") | (rfm["segment"]=="loyal_customers")])

women_cat = df[(df["interested_in_categories_12"]).str.contains("KADIN")]

women_vip_cust = pd.merge(vip_cust,women_cat[["interested_in_categories_12","master_id"]],on=["master_id"])


pd.concat(df[(df["interested_in_categories_12"]).str.contains("KADIN")],df[(df["interested_in_categories_12"]).str.contains("KADIN")])
women_vip_cust.columns
# Index(['master_id', 'Recency', 'Frequency', 'Monetary', 'Recency_score',
#        'Frequency_score', 'Monetary_score', 'RFM_score', 'segment',
#        'interested_in_categories_12'],
#       dtype='object')

women_vip_cust_ = women_vip_cust.drop(women_vip_cust.loc[:, 'Recency':'interested_in_categories_12'].columns, axis=1)

women_vip_cust_.to_csv("wom.csv")
women_vip_cust_.to_csv("women_vip_customer_info.csv")



##############################################################################################
#                                      Example Case                                          #
##############################################################################################
# For the men and children category, customers who have not been shopping for a long time, those who are asleep,
# and new customers should not be lost. It is desired to be specially selected for the customers with this feature.
# It is planned to apply a 40% discount to these customers.
cus_profile = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="about_to_sleep") | (rfm["segment"]=="new_customers")]
man_boy_cus =df[(df["interested_in_categories_12"]).str.contains("ERKEK") | (df["interested_in_categories_12"]).str.contains("COCUK")]
man_boy_cus_profile = pd.merge(cus_profile,man_boy_cus[["interested_in_categories_12","master_id"]],on=["master_id"])
man_boy_cus_profile = man_boy_cus_profile.drop(man_boy_cus_profile.loc[:,'Recency':'interested_in_categories_12'].columns, axis=1)
man_boy_cus_profile
man_boy_cus_profile.to_csv("man_boy_customer_profile.csv")
############################################################################################

rfm1 = df.groupby('master_id').agg( {'last_order_date': lambda last_order_date : (today_date- last_order_date.max()).days,
                                    'order_num_total_online_offline': lambda total_purchases : total_purchases.sum(),
                                    'customer_onoff_total_expense': lambda total_expense: total_expense.sum(),
                                    })

rfm1.columns = ['Recency', 'Frequency', 'Monetary']
rfm1['Recency_score'] = pd.qcut(rfm1['Recency'],5, labels = [5,4,3,2,1] )
rfm1["Frequency_score"] = pd.qcut(rfm1['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm1["Monetary_score"] = pd.qcut(rfm1['Monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm1['RFM_score'] = (rfm1['Recency_score'].astype(str)+ rfm1['Frequency_score'].astype(str)+rfm1['Monetary_score'].astype(str) )

seg_map = {
    r'5[1-2][1-2]': 'new_customer',
    r'[1-2][1-2][1-2]': 'lost_segment',
    r'[1-2][3-5][3-5]':'cant_loose',
    r'[2-3][3-4][2-4]': 'awake_segment',
    r'[3-4][3-4][3-4]': 'potential_segment',
    r'[4][4-5][4-5]': 'loyal_segment',
    r'5[4-5][4-5]': 'top_segment'
}

seg_map

rfm1['segment'] = rfm1['RFM_score'].replace(seg_map, regex=True)
rfm1[["segment", "RFM_score"]].groupby("segment").agg(["mean", "count"])
rfm1[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])
rfm1['segment'].value_counts().head(20)
