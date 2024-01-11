

# CUSTOMER SEGMENTATION WITH RFM ANALYSIS PROJECT

# BUSINESS PROBLEM:

# FLO,an online shoe store,wants to segment its customers and determine marketing strategies
# according to these segments. In this regard, the behavior of customers will be defined and
# groups will be formed according to the clustering in these behaviors.



# Story of Dataset

# The dataset shows the customers which last purchases from the FLO store on Omnichannel(both online and offline
# shopping store)in 2020-2021. However, these customers have consist of infomation from their past shopping behavior.



# master_id : Unique Customer Number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile))
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : The customer's first purchase date
# last_order_date : The customer's last purchase date
# last_order_date_online :  The customer's last purchase date in online shopping platform
# last_order_date_offline : The customer's last purchase date in offline shopping platform
# order_num_total_ever_online : The customer's total purchases in online shopping platform
# order_num_total_ever_offline :  The customer's total purchases in offline shopping platform
# customer_value_total_ever_offline : The total expenditure by customer in offline shopping platform
# customer_value_total_ever_online : The total expenditure by customer in online shopping platform
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months



##############################################################################################
#                          MISSION 1 : DATA UNDERSTANDING AND PREPARATION                    #
##############################################################################################

# STEP 1 - IMPORT LIBRARIES

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

# STEP 2

df.head(10)
# Out[79]:
#                               master_id order_channel last_order_channel  \
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile
# 2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App
# 3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f       Desktop            Offline
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f   Android App        Android App
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f        Mobile            Offline
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f   Android App        Android App
# 9  1143f032-440d-11ea-8b43-000d3a38a36f        Mobile             Mobile
#   first_order_date last_order_date last_order_date_online  \
# 0       2020-10-30      2021-02-26             2021-02-21
# 1       2017-02-08      2021-02-16             2021-02-16
# 2       2019-11-27      2020-11-27             2020-11-27
# 3       2021-01-06      2021-01-17             2021-01-17
# 4       2019-08-03      2021-03-07             2021-03-07
# 5       2018-11-18      2021-03-13             2018-11-18
# 6       2020-03-04      2020-10-18             2020-10-18
# 7       2020-05-15      2020-08-12             2020-05-15
# 8       2020-01-23      2021-03-07             2021-03-07
# 9       2019-07-30      2020-10-04             2020-10-04
#  last_order_date_offline  order_num_total_ever_online  \
# 0              2021-02-26                        4.000
# 1              2020-01-10                       19.000
# 2              2019-12-01                        3.000
# 3              2021-01-06                        1.000
# 4              2019-08-03                        1.000
# 5              2021-03-13                        1.000
# 6              2020-03-04                        3.000
# 7              2020-08-12                        1.000
# 8              2020-01-25                        3.000
# 9              2019-07-30                        1.000
#    order_num_total_ever_offline  customer_value_total_ever_offline  \
# 0                         1.000                            139.990
# 1                         2.000                            159.970
# 2                         2.000                            189.970
# 3                         1.000                             39.990
# 4                         1.000                             49.990
# 5                         2.000                            150.870
# 6                         1.000                             59.990
# 7                         1.000                             49.990
# 8                         2.000                            120.480
# 9                         1.000                             69.980
#   customer_value_total_ever_online       interested_in_categories_12
# 0                           799.380                           [KADIN]
# 1                          1853.580  [ERKEK, COCUK, KADIN, AKTIFSPOR]
# 2                           395.350                    [ERKEK, KADIN]
# 3                            81.980               [AKTIFCOCUK, COCUK]
# 4                           159.990                       [AKTIFSPOR]
# 5                            49.990                           [KADIN]
# 6                           315.940                       [AKTIFSPOR]
# 7                           113.640                           [COCUK]
# 8                           934.210             [ERKEK, COCUK, KADIN]
# 9                            95.980                [KADIN, AKTIFSPOR]


# Değişken İsimleri :
df.columns
# Out[80]:
# Index(['master_id', 'order_channel', 'last_order_channel', 'first_order_date',
#        'last_order_date', 'last_order_date_online', 'last_order_date_offline',
#        'order_num_total_ever_online', 'order_num_total_ever_offline',
#        'customer_value_total_ever_offline', 'customer_value_total_ever_online',
#        'interested_in_categories_12'],
#       dtype='object')

df.interested_in_categories_12.value_counts()
# [AKTIFSPOR]                                     3464
# [KADIN]                                         2158
# []                                              2135
# [ERKEK]                                         1973
# [KADIN, AKTIFSPOR]                              1352
# [ERKEK, AKTIFSPOR]                              1178
# [ERKEK, KADIN]                                   848
# [COCUK]                                          836
# [ERKEK, KADIN, AKTIFSPOR]                        775
# [AKTIFCOCUK]                                     679
# [COCUK, KADIN]                                   443
# [AKTIFCOCUK, COCUK]                              349
# [AKTIFCOCUK, AKTIFSPOR]                          317
# [COCUK, AKTIFSPOR]                               317
# [COCUK, KADIN, AKTIFSPOR]                        241
# [AKTIFCOCUK, ERKEK, COCUK, KADIN, AKTIFSPOR]     223
# [ERKEK, COCUK]                                   215
# [ERKEK, COCUK, KADIN, AKTIFSPOR]                 213
# [AKTIFCOCUK, COCUK, KADIN]                       213
# [AKTIFCOCUK, KADIN]                              210
# [ERKEK, COCUK, KADIN]                            204
# [AKTIFCOCUK, COCUK, KADIN, AKTIFSPOR]            203
# [AKTIFCOCUK, COCUK, AKTIFSPOR]                   202
# [AKTIFCOCUK, KADIN, AKTIFSPOR]                   184
# [ERKEK, COCUK, AKTIFSPOR]                        156
# [AKTIFCOCUK, ERKEK]                              152
# [AKTIFCOCUK, ERKEK, AKTIFSPOR]                   142
# [AKTIFCOCUK, ERKEK, KADIN, AKTIFSPOR]            132
# [AKTIFCOCUK, ERKEK, COCUK]                       122
# [AKTIFCOCUK, ERKEK, COCUK, KADIN]                115
# [AKTIFCOCUK, ERKEK, COCUK, AKTIFSPOR]            105
# [AKTIFCOCUK, ERKEK, KADIN]                        89
# Name: interested_in_categories_12, dtype: int64


df.order_channel.value_counts()
# Android App    9495
# Mobile         4882
# Ios App        2833
# Desktop        2735
# Name: order_channel, dtype: int64


df.nunique()
# master_id                            19945
# order_channel                            4
# last_order_channel                       5
# first_order_date                      2465
# last_order_date                        366
# last_order_date_online                1743
# last_order_date_offline                738
# order_num_total_ever_online             57
# order_num_total_ever_offline            32
# customer_value_total_ever_offline     6097
# customer_value_total_ever_online     11292
# interested_in_categories_12             32
# order_num_total_online_offline          63
# customer_onoff_total_expense         16277


# Betimsel İstatistik
df.describe().T
#                                       count    mean     std    min     25%  \
# order_num_total_ever_online       19945.000   3.111   4.226  1.000   1.000
# order_num_total_ever_offline      19945.000   1.914   2.063  1.000   1.000
# customer_value_total_ever_offline 19945.000 253.923 301.533 10.000  99.990
# customer_value_total_ever_online  19945.000 497.322 832.602 12.990 149.980
#                                       50%     75%       max
# order_num_total_ever_online         2.000   4.000   200.000
# order_num_total_ever_offline        1.000   2.000   109.000
# customer_value_total_ever_offline 179.980 319.970 18119.140
# customer_value_total_ever_online  286.460 578.440 45220.130


# Boş Değerler
df.isnull().sum()
# Out[82]:
# master_id                            0
# order_channel                        0
# last_order_channel                   0
# first_order_date                     0
# last_order_date                      0
# last_order_date_online               0
# last_order_date_offline              0
# order_num_total_ever_online          0
# order_num_total_ever_offline         0
# customer_value_total_ever_offline    0
# customer_value_total_ever_online     0
# interested_in_categories_12          0
# dtype: int64


# Değişken Tipleri
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 19945 entries, 0 to 19944
# Data columns (total 12 columns):
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   master_id                          19945 non-null  object
#  1   order_channel                      19945 non-null  object
#  2   last_order_channel                 19945 non-null  object
#  3   first_order_date                   19945 non-null  object
#  4   last_order_date                    19945 non-null  object
#  5   last_order_date_online             19945 non-null  object
#  6   last_order_date_offline            19945 non-null  object
#  7   order_num_total_ever_online        19945 non-null  float64
#  8   order_num_total_ever_offline       19945 non-null  float64
#  9   customer_value_total_ever_offline  19945 non-null  float64
#  10  customer_value_total_ever_online   19945 non-null  float64
#  11  interested_in_categories_12        19945 non-null  object
# dtypes: float64(4), object(8)
# memory usage: 1.8+ MB

# Function of Checking Dataframe and Missing Values

# Summary of Dataset Variables

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

# --------------------- Shape ---------------------
# (19945, 14)
# --------------------- Types ---------------------
# master_id                                    object
# order_channel                                object
# last_order_channel                           object
# first_order_date                     datetime64[ns]
# last_order_date                      datetime64[ns]
# last_order_date_online               datetime64[ns]
# last_order_date_offline              datetime64[ns]
# order_num_total_ever_online                 float64
# order_num_total_ever_offline                float64
# customer_value_total_ever_offline           float64
# customer_value_total_ever_online            float64
# interested_in_categories_12                  object
# order_num_total_online_offline              float64
# customer_onoff_total_expense                float64
# dtype: object
# --------------------- Head ---------------------
#                               master_id order_channel last_order_channel  \
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile
# 2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App
# 3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f       Desktop            Offline
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f   Android App        Android App
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f        Mobile            Offline
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f   Android App        Android App
# 9  1143f032-440d-11ea-8b43-000d3a38a36f        Mobile             Mobile
#   first_order_date last_order_date last_order_date_online  \
# 0       2020-10-30      2021-02-26             2021-02-21
# 1       2017-02-08      2021-02-16             2021-02-16
# 2       2019-11-27      2020-11-27             2020-11-27
# 3       2021-01-06      2021-01-17             2021-01-17
# 4       2019-08-03      2021-03-07             2021-03-07
# 5       2018-11-18      2021-03-13             2018-11-18
# 6       2020-03-04      2020-10-18             2020-10-18
# 7       2020-05-15      2020-08-12             2020-05-15
# 8       2020-01-23      2021-03-07             2021-03-07
# 9       2019-07-30      2020-10-04             2020-10-04
#   last_order_date_offline  order_num_total_ever_online  \
# 0              2021-02-26                        4.000
# 1              2020-01-10                       19.000
# 2              2019-12-01                        3.000
# 3              2021-01-06                        1.000
# 4              2019-08-03                        1.000
# 5              2021-03-13                        1.000
# 6              2020-03-04                        3.000
# 7              2020-08-12                        1.000
# 8              2020-01-25                        3.000
# 9              2019-07-30                        1.000
#    order_num_total_ever_offline  customer_value_total_ever_offline  \
# 0                         1.000                            139.990
# 1                         2.000                            159.970
# 2                         2.000                            189.970
# 3                         1.000                             39.990
# 4                         1.000                             49.990
# 5                         2.000                            150.870
# 6                         1.000                             59.990
# 7                         1.000                             49.990
# 8                         2.000                            120.480
# 9                         1.000                             69.980
#    customer_value_total_ever_online       interested_in_categories_12  \
# 0                           799.380                           [KADIN]
# 1                          1853.580  [ERKEK, COCUK, KADIN, AKTIFSPOR]
# 2                           395.350                    [ERKEK, KADIN]
# 3                            81.980               [AKTIFCOCUK, COCUK]
# 4                           159.990                       [AKTIFSPOR]
# 5                            49.990                           [KADIN]
# 6                           315.940                       [AKTIFSPOR]
# 7                           113.640                           [COCUK]
# 8                           934.210             [ERKEK, COCUK, KADIN]
# 9                            95.980                [KADIN, AKTIFSPOR]
#    order_num_total_online_offline  customer_onoff_total_expense
# 0                           5.000                       939.370
# 1                          21.000                      2013.550
# 2                           5.000                       585.320
# 3                           2.000                       121.970
# 4                           2.000                       209.980
# 5                           3.000                       200.860
# 6                           4.000                       375.930
# 7                           2.000                       163.630
# 8                           5.000                      1054.690
# 9                           2.000                       165.960
# --------------------- Missing Values Analysis ---------------------
# Empty DataFrame
# Columns: [Total Missing Values, Ratio]
# Index: []
# --------------------- Quantiles ---------------------
#                                    0.000   0.050   0.500    0.950    0.990  \
# order_num_total_ever_online        1.000   1.000   2.000   10.000   20.000
# order_num_total_ever_offline       1.000   1.000   1.000    4.000    7.000
# customer_value_total_ever_offline 10.000  39.990 179.980  694.222 1219.947
# customer_value_total_ever_online  12.990  63.990 286.460 1556.726 3143.810
# order_num_total_online_offline     2.000   2.000   4.000   12.000   22.000
# customer_onoff_total_expense      44.980 175.480 545.270 1921.924 3606.356
#                                       1.000
# order_num_total_ever_online         200.000
# order_num_total_ever_offline        109.000
# customer_value_total_ever_offline 18119.140
# customer_value_total_ever_online  45220.130
# order_num_total_online_offline      202.000
# customer_onoff_total_expense      45905.100

# STEP 3 # Total Purchases Number and Total Expense for Omnichannel Customers

# Total purchases for omnichannel customers
df["order_num_total_online_offline"] = df["order_num_total_ever_online"] +  df["order_num_total_ever_offline"]

# Total expense for omnichannel customers
df["customer_onoff_total_expense"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# STEP 4 Review the data type for variable expressing the date

df.info()
#  3   first_order_date                   19945 non-null  object
#  4   last_order_date                    19945 non-null  object
#  5   last_order_date_online             19945 non-null  object
#  6   last_order_date_offline            19945 non-null  object

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
#  3   first_order_date                   19945 non-null  datetime64[ns]
#  4   last_order_date                    19945 non-null  datetime64[ns]
#  5   last_order_date_online             19945 non-null  datetime64[ns]
#  6   last_order_date_offline            19945 non-null  datetime64[ns]


# STEP 5 #
# Distribution of the number of Customers in the Shopping Channels,
# the total number of products purchased and their total expenditures
df.groupby('order_channel').agg({'order_num_total_online_offline': 'sum',
                                'customer_onoff_total_expense':'count'}).sort_values(by="customer_onoff_total_expense",ascending=False)


df.groupby('order_channel').agg({'order_num_total_online_offline': ['count','sum'],
                                'customer_onoff_total_expense':['count','sum']})
#               order_num_total_online_offline             customer_onoff_total_expense
#                                        count         sum                        count           sum
# order_channel
# Android App                             9495 52269.00000                         9495 7819062.76000
# Desktop                                 2735 10920.00000                         2735 1610321.46000
# Ios App                                 2833 15351.00000                         2833 2525999.93000
# Mobile                                  4882 21679.00000                         4882 3028183.16000


df["master_id"].nunique()

################## Bonus #############################################################################3
df.groupby('last_order_channel').agg({'order_num_total_online_offline': 'sum',
                                'customer_onoff_total_expense':'count'}).sort_values(by="customer_onoff_total_expense",ascending=False)
#                     order_num_total_online_offline  customer_onoff_total_expense
# last_order_channel
# Android App                            37320.00000                          6783
# Offline                                30643.00000                          6608
# Mobile                                 14195.00000                          3172
# Ios App                                 8595.00000                          1696
# Desktop                                 9466.00000                          1686


# STEP 6 # Top 10 Customers with the Most Profits
df.groupby('master_id').agg({'customer_onoff_total_expense':'sum'}).\
    sort_values(by="customer_onoff_total_expense",ascending=False).head(10)
#                                       customer_onoff_total_expense
# master_id
# 5d1c466a-9cfd-11e9-9897-000d3a38a36f                     45905.100
# d5ef8058-a5c6-11e9-a2fc-000d3a38a36f                     36818.290
# 73fd19aa-9e37-11e9-9897-000d3a38a36f                     33918.100
# 7137a5c0-7aad-11ea-8f20-000d3a38a36f                     31227.410
# 47a642fe-975b-11eb-8c2a-000d3a38a36f                     20706.340
# a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                     18443.570
# d696c654-2633-11ea-8e1c-000d3a38a36f                     16918.570
# fef57ffa-aae6-11e9-a2fc-000d3a38a36f                     12726.100
# cba59206-9dd1-11e9-9897-000d3a38a36f                     12282.240
# fc0ce7a4-9d87-11e9-9897-000d3a38a36f                     12103.150



# STEP 7 # Top 10 Customers with Most Orders
df.groupby('master_id').agg({'order_num_total_online_offline':'sum'}).\
    sort_values(by="order_num_total_online_offline",ascending=False).head(10)
#                                       order_num_total_online_offline
# master_id
# 5d1c466a-9cfd-11e9-9897-000d3a38a36f                         202.000
# cba59206-9dd1-11e9-9897-000d3a38a36f                         131.000
# a57f4302-b1a8-11e9-89fa-000d3a38a36f                         111.000
# fdbe8304-a7ab-11e9-a2fc-000d3a38a36f                          88.000
# 329968c6-a0e2-11e9-a2fc-000d3a38a36f                          83.000
# 73fd19aa-9e37-11e9-9897-000d3a38a36f                          82.000
# 44d032ee-a0d4-11e9-a2fc-000d3a38a36f                          77.000
# b27e241a-a901-11e9-a2fc-000d3a38a36f                          75.000
# d696c654-2633-11ea-8e1c-000d3a38a36f                          70.000
# a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                          70.000

# Bonus

df.groupby('master_id').agg({'order_num_total_online_offline': 'sum',
                                'customer_onoff_total_expense':'sum'}).sort_values(by="customer_onoff_total_expense",ascending=False).head(10)
#                                       order_num_total_online_offline  customer_onoff_total_expense
# master_id
# 5d1c466a-9cfd-11e9-9897-000d3a38a36f                       202.00000                   45905.10000
# d5ef8058-a5c6-11e9-a2fc-000d3a38a36f                        68.00000                   36818.29000
# 73fd19aa-9e37-11e9-9897-000d3a38a36f                        82.00000                   33918.10000
# 7137a5c0-7aad-11ea-8f20-000d3a38a36f                        11.00000                   31227.41000
# 47a642fe-975b-11eb-8c2a-000d3a38a36f                         4.00000                   20706.34000
# a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                        70.00000                   18443.57000
# d696c654-2633-11ea-8e1c-000d3a38a36f                        70.00000                   16918.57000
# fef57ffa-aae6-11e9-a2fc-000d3a38a36f                        37.00000                   12726.10000
# cba59206-9dd1-11e9-9897-000d3a38a36f                       131.00000                   12282.24000
# fc0ce7a4-9d87-11e9-9897-000d3a38a36f                        20.00000                   12103.15000


# En yüksek satınalma yapan müşterimiz, bizim en fazla kazanç getiren müşterimiz midir?
# Sıralamayı baz aldığımızda ilk 5 müşteriyi analiz ettiğimizde çıkan sonuçları değerlendirebilir miyiz?
# Neden RFM yapmaktayız?


# Step 8 # Data Preparation Process Function
def data_preparation(dataframe):
    dataframe["order_num_total_online_offline"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]

    dataframe["customer_onoff_total_expense"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    dataframe[dates] = dataframe[dates].apply(pd.to_datetime)

    return dataframe

data_preparation(df)

### BONUS
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
#                          MISSION 1 : CALCULATING RFM METRICS                               #
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
#                                       Recency  Frequency  Monetary
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f       11      5.000   776.070
# 00034aaa-a838-11e9-a2fc-000d3a38a36f      299      3.000   269.470
# 000be838-85df-11ea-a90b-000d3a38a36f      214      4.000   722.690
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f       28      7.000   874.160
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f       21      7.000  1620.330
# 00136ce2-a562-11e9-a2fc-000d3a38a36f      204      2.000   359.450
# 00142f9a-7af6-11eb-8460-000d3a38a36f       26      3.000   404.940
# 0014778a-5b11-11ea-9a2c-000d3a38a36f       27      3.000   727.430
# 0018c6aa-ab6c-11e9-a2fc-000d3a38a36f      127      2.000   317.910
# 0022f41e-5597-11eb-9e65-000d3a38a36f       13      2.000   154.980

##############################################################################################
#                          MISSION 3 : CALCULATING RFM SCORES                                #
##############################################################################################

# MISSION 3 CALCULATING RFM SCORES

# RFM değeri
rfm.describe()


rfm['Recency_score'] = pd.qcut(rfm['Recency'],5, labels = [5,4,3,2,1] )
rfm["Frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()
#                                       Recency  Frequency  Monetary  \
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f       11      5.000   776.070
# 00034aaa-a838-11e9-a2fc-000d3a38a36f      299      3.000   269.470
# 000be838-85df-11ea-a90b-000d3a38a36f      214      4.000   722.690
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f       28      7.000   874.160
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f       21      7.000  1620.330
#                                      Recency_score Frequency_score  \
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f             5               4
# 00034aaa-a838-11e9-a2fc-000d3a38a36f             1               2
# 000be838-85df-11ea-a90b-000d3a38a36f             2               3
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f             5               4
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f             5               4
#                                      Monetary_score
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f              4
# 00034aaa-a838-11e9-a2fc-000d3a38a36f              1
# 000be838-85df-11ea-a90b-000d3a38a36f              4
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f              4
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f              5


rfm['RFM_score'] = (rfm['Recency_score'].astype(str)+ rfm['Frequency_score'].astype(str))

rfm[rfm["RFM_score"] == "55"].head(10)
#                                       Recency  Frequency  Monetary  \
# master_id
# 004d5204-2037-11ea-87bf-000d3a38a36f       28      8.000  1170.760
# 00736820-a834-11e9-a2fc-000d3a38a36f       27      9.000   714.530
# 00b3ee24-aa44-11e9-a2fc-000d3a38a36f       25      8.000  2027.780
# 00cf8494-9da2-11e9-9897-000d3a38a36f        6     53.000  6275.330
# 0151bbee-a7de-11e9-a2fc-000d3a38a36f       15     18.000  2649.020
# 016521aa-aa88-11e9-a2fc-000d3a38a36f       28     10.000  1691.280
# 020fdc82-a8d3-11e9-a2fc-000d3a38a36f        7     10.000  1735.080
# 024da65a-5b36-11ea-b7e2-000d3a38a36f       16     11.000  2312.010
# 025b8bb6-ac28-11e9-a2fc-000d3a38a36f        7     11.000   820.880
# 02a5bc6c-d663-11e9-93bc-000d3a38a36f       20     11.000  1578.700
#                                      Recency_score Frequency_score  \
# master_id
# 004d5204-2037-11ea-87bf-000d3a38a36f             5               5
# 00736820-a834-11e9-a2fc-000d3a38a36f             5               5
# 00b3ee24-aa44-11e9-a2fc-000d3a38a36f             5               5
# 00cf8494-9da2-11e9-9897-000d3a38a36f             5               5
# 0151bbee-a7de-11e9-a2fc-000d3a38a36f             5               5
# 016521aa-aa88-11e9-a2fc-000d3a38a36f             5               5
# 020fdc82-a8d3-11e9-a2fc-000d3a38a36f             5               5
# 024da65a-5b36-11ea-b7e2-000d3a38a36f             5               5
# 025b8bb6-ac28-11e9-a2fc-000d3a38a36f             5               5
# 02a5bc6c-d663-11e9-93bc-000d3a38a36f             5               5
#                                      Monetary_score RFM_score
# master_id
# 004d5204-2037-11ea-87bf-000d3a38a36f              5        55
# 00736820-a834-11e9-a2fc-000d3a38a36f              4        55
# 00b3ee24-aa44-11e9-a2fc-000d3a38a36f              5        55
# 00cf8494-9da2-11e9-9897-000d3a38a36f              5        55
# 0151bbee-a7de-11e9-a2fc-000d3a38a36f              5        55
# 016521aa-aa88-11e9-a2fc-000d3a38a36f              5        55
# 020fdc82-a8d3-11e9-a2fc-000d3a38a36f              5        55
# 024da65a-5b36-11ea-b7e2-000d3a38a36f              5        55
# 025b8bb6-ac28-11e9-a2fc-000d3a38a36f              4        55

##############################################################################################
#                          MISSION 4 : DEFINING RFM SCORE AS A SEGMENT                       #
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
#                     Recency       Frequency       Monetary
#                        mean count      mean count     mean count
# segment
# about_to_sleep      115.032  1643     2.407  1643  361.649  1643
# at_Risk             243.329  3152     4.470  3152  648.325  3152
# cant_loose          236.159  1194    10.717  1194 1481.652  1194
# champions            18.142  1920     8.965  1920 1410.709  1920
# hibernating         248.426  3589     2.391  3589  362.583  3589
# loyal_customers      83.558  3375     8.356  3375 1216.257  3375
# need_attention      114.037   806     3.739   806  553.437   806
# new_customers        18.976   673     2.000   673  344.049   673
# potential_loyalists  37.870  2925     3.311  2925  533.741  2925
# promising            59.695   668     2.000   668  334.153   668


rfm[rfm["segment"] == "need_attention"].head(10)
#                                       Recency  Frequency  Monetary  \
# master_id
# 0033a502-5bf1-11ea-829b-000d3a38a36f      106      4.000   788.730
# 003c4ebc-aa23-11e9-a2fc-000d3a38a36f      109      4.000   360.760
# 00f53518-ab9e-11e9-a2fc-000d3a38a36f       89      4.000   349.940
# 012fe082-b134-11e9-9757-000d3a38a36f      137      4.000   609.940
# 019443fe-ab05-11e9-a2fc-000d3a38a36f       89      4.000   317.450
# 023db43a-aa05-11e9-a2fc-000d3a38a36f      100      4.000   489.870
# 02bf2f16-ad15-11e9-a2fc-000d3a38a36f      100      4.000   330.850
# 030b5cca-ab25-11e9-a2fc-000d3a38a36f       92      4.000   477.950
# 0393a9d8-541f-11ea-b1db-000d3a38a36f      100      4.000   535.900
# 03b39e0a-e205-11e9-957d-000d3a38a36f      115      4.000   429.960
#                                      Recency_score Frequency_score  \
# master_id
# 0033a502-5bf1-11ea-829b-000d3a38a36f             3               3
# 003c4ebc-aa23-11e9-a2fc-000d3a38a36f             3               3
# 00f53518-ab9e-11e9-a2fc-000d3a38a36f             3               3
# 012fe082-b134-11e9-9757-000d3a38a36f             3               3
# 019443fe-ab05-11e9-a2fc-000d3a38a36f             3               3
# 023db43a-aa05-11e9-a2fc-000d3a38a36f             3               3
# 02bf2f16-ad15-11e9-a2fc-000d3a38a36f             3               3
# 030b5cca-ab25-11e9-a2fc-000d3a38a36f             3               3
# 0393a9d8-541f-11ea-b1db-000d3a38a36f             3               3
# 03b39e0a-e205-11e9-957d-000d3a38a36f             3               3
#                                      Monetary_score RFM_score         segment
# master_id
# 0033a502-5bf1-11ea-829b-000d3a38a36f              4        33  need_attention
# 003c4ebc-aa23-11e9-a2fc-000d3a38a36f              2        33  need_attention
# 00f53518-ab9e-11e9-a2fc-000d3a38a36f              2        33  need_attention
# 012fe082-b134-11e9-9757-000d3a38a36f              3        33  need_attention
# 019443fe-ab05-11e9-a2fc-000d3a38a36f              2        33  need_attention
# 023db43a-aa05-11e9-a2fc-000d3a38a36f              3        33  need_attention
# 02bf2f16-ad15-11e9-a2fc-000d3a38a36f              2        33  need_attention
# 030b5cca-ab25-11e9-a2fc-000d3a38a36f              3        33  need_attention
# 0393a9d8-541f-11ea-b1db-000d3a38a36f              3        33  need_attention
# 03b39e0a-e205-11e9-957d-000d3a38a36f              2        33  need_attention

rfm.head()
rfm.info()
## Write a csv file to a new folder

from pathlib import Path
filepath = Path('D:/12thTerm_DS_Bootcamp/3Week_CRM_Analytics/rfm.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
rfm.to_csv(filepath)


##############################################################################################
#                                          CASES 1                                           #
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
#                                           CASES 2                                          #
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



############################################################################################3

# BONUS #

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