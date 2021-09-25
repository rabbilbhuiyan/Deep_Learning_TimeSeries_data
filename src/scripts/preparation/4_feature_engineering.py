# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt # display figure inline in notebook
#import plotly.express as px # view the timeseries data in a slider
import seaborn as sns
#sns.set(rc={'figure.figsize':(8,5)}) # set the seaborn default figrue size
sns.set()
sns.set_style('white') #possible choices: white, dark, whitegrid, darkgrid, ticks
sns.set(rc={'figure.figsize':(8,5)}, style="white", font_scale=1.4)#style="whitegrid"
plt.rcParams["axes.labelsize"] = 12

from datetime import datetime # converting the timestamp data into datetime object
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # determining autocorrelation
from scipy import stats # detecting outliers by z score

# +
SAVING_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\outputs\Vizualization'

DATA_PATH = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed\outlier_replace_processed_data.csv'
df = pd.read_csv(DATA_PATH)
df.head(4)
# -

df.shape

# dropping few columns that are not necessary at this stage
df= df.drop(['group', 'spora-id', 'latitude', 'longitude'], axis=1)

# +
# converting the timestamp data into datetime object 
df['date'] = pd.to_datetime(df['date'])

# setting index
df.set_index(['date'], inplace = True)

# resmaple to hourly frequency, aggregating with mean
df_by_hour = df.resample('H').mean()

print(df_by_hour.head())
print(df_by_hour.shape)

# +
# filling the NaN value using ffill method
df_by_hour.fillna(method='ffill', inplace=True)

# Checking the number of null value
df_by_hour.isnull().sum()
# -

# ## Feature Engineering
# ### Temporal feature engineering

# +
# Feature engineering with timestamp data
# Extracting year, month and day information
# We can do it easily as DatetimeIndex has individual attributes for year, month, day etc

# Add columns with year, month, day and weeday name
df_by_hour['Hour'] = df_by_hour.index.hour
df_by_hour['Day_of_week'] = df_by_hour.index.dayofweek
df_by_hour['Day_of_month'] = df_by_hour.index.day
df_by_hour['Month'] = df_by_hour.index.month
df_by_hour['Day'] = df_by_hour.index.day_name()

# create the feature of 'is_weekday' from Day column
# Defining the day as weekday (1) or weekend (0)
filter1 = df_by_hour['Day']== 'Saturday'
filter2 = df_by_hour['Day']== 'Sunday'

df_by_hour['is_weekday']= np.where(filter1 | filter2, 0, 1)

df_by_hour.head(2)


# +
# let's see the value for Saturday and Sunday
#df_by_hour.query("Day=='Saturday' or Day =='Sunday'")
# -

# Day part
# creating four groups of daypart on Hour column
# daypart function
def daypart(hour):
    if hour in [4,5,6,7,8,9]:
        return "Morning"
    elif hour in [10,11,12,13,14,15]:
        return "Noon"
    elif hour in [16,17,18,19,20,21]:
        return "Evening"
    else: return "Night"
# utilize it along with apply method
df_by_hour['daypart'] = df_by_hour.Hour.apply(daypart)

# converting categorical variable into numerical
# alternate option: train['city'] = train['city'].astype('category').cat.codes
df_by_hour['daypart'] = pd.factorize(df_by_hour.daypart)[0]


# creating season
def season(month):
    if month in [2,3,4]:
        return "Spring"
    else: return "Summer"
# utilize it along with apply method
df_by_hour['Season'] = df_by_hour.Month.apply(season)

# +
# converting categorical variable into numerical
# alternate option: train['city'] = train['city'].astype('category').cat.codes
df_by_hour['Season'] = pd.factorize(df_by_hour.Season)[0]

# printing df
df_by_hour.tail(2)
# -

# #### Vizualizing the data after feature engineering
# - Let's see how our feature engineering effort work, how well the new features separate the data

# +
#df_by_month = df_by_hour.resample('M').mean()
#sns.lineplot(x=df_by_month.index, y='O3', data=df_by_month)

# Monthly pattern of data
sns.lineplot(x='Month', y='NO', data=df_by_hour.ffill())
# -

# Hourly pattern of data
sns.pointplot(x='Hour', y='NO', data=df_by_hour.ffill())

# Day of week pattern 
sns.pointplot(x='Day_of_week', y='NO', data=df_by_hour.ffill())

sns.pointplot(x='Hour', y='NO', data=df_by_hour.ffill(), hue='is_weekday')

sns.pointplot(x='Hour', y='NO', data=df_by_hour.ffill(), hue='daypart')

sns.lineplot(x='Month', y='NO', hue='Season', data=df_by_hour.ffill())

# ### Statistical feature engineering

# +
# plot_acf is to idenfify parameter for moving average 
#plot_acf(df_by_hour.resample('D').mean().ffill().O3)# to identify the value of moving average
# we choosed 5 day rolling mean or 120 hour rolling mean
# however, for o3 it should be 10 days
#plt.savefig('autoC_O3.png')

# +
import statsmodels.api as sm
import matplotlib.patches as mpatches # for manually adding legend

def plot_acf_pacf(df,col1, label_name, file_name):
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    #df = df_by_hour.copy()
    
    patch = mpatches.Patch(label=label_name)
    plt.legend(handles=[patch])  
    
    axes = sm.graphics.tsa.plot_acf(df[col1].resample('12H').mean().ffill(), lags=30, ax=ax[0])
    patch = mpatches.Patch(label=label_name)
    plt.legend(handles=[patch])  

    axes = sm.graphics.tsa.plot_pacf(df[col1].resample('H').mean().ffill(), lags=30,ax=ax[1])
    patch = mpatches.Patch(label=label_name)
    plt.legend(handles=[patch])  
    
    
    my_path = SAVING_DIR
    my_file = file_name
    plt.savefig(os.path.join(my_path, my_file))
    

plot_acf_pacf(df_by_hour,col1='NO', label_name='NO', file_name='NO_auto_partial_correlation.png')
plot_acf_pacf(df_by_hour,col1='NO2', label_name='NO2', file_name='NO2_auto_partial_correlation.png')
plot_acf_pacf(df_by_hour,col1='CO', label_name='CO', file_name='CO_auto_partial_correlation.png')
plot_acf_pacf(df_by_hour,col1='O3', label_name='O3', file_name='O3_auto_partial_correlation.png')
# -

'''
# to identify the value of autoregression (lag number)
# we see that after 4 observation there is no corelation (which is in critical level) so we seletct order of 4 or 5 
# so we found the lag value differnece as 7 for NO, NO2, 8 for CO, 20 for o3 (x12) ;
# however that of pacf is 4 hours for all concentrati
'''

# +
# creating more feautres based on lagged value difference 
df_by_hour['NO_lag_feature']= df_by_hour.NO.shift(4)
df_by_hour['NO2_lag_feature']= df_by_hour.NO2.shift(4)
df_by_hour['CO_lag_feature']= df_by_hour.CO.shift(4)
df_by_hour['O3_lag_feature']= df_by_hour.O3.shift(4)

# creating more features based on rolling mean
df_by_hour['NO_rolling_mean']= df_by_hour.NO.rolling(window=84).mean()
df_by_hour['NO2_rolling_mean']= df_by_hour.NO2.rolling(window=84).mean()
df_by_hour['CO_rolling_mean']= df_by_hour.CO.rolling(window=96).mean()
df_by_hour['O3_rolling_mean']= df_by_hour.O3.rolling(window=240).mean()

# creating more features based on rolling min
df_by_hour['NO_rolling_min']= df_by_hour.NO.rolling(window=84).min()
df_by_hour['NO2_rolling_min']= df_by_hour.NO2.rolling(window=84).min()
df_by_hour['CO_rolling_min']= df_by_hour.CO.rolling(window=96).min()
df_by_hour['O3_rolling_min']= df_by_hour.O3.rolling(window=240).min()

# creating more features based on rolling max
df_by_hour['NO_rolling_max']= df_by_hour.NO.rolling(window=84).max()
df_by_hour['NO2_rolling_max']= df_by_hour.NO2.rolling(window=84).max()
df_by_hour['CO_rolling_max']= df_by_hour.CO.rolling(window=96).max()
df_by_hour['O3_rolling_max']= df_by_hour.O3.rolling(window=240).max()

# creating more features based on rolling deviation 
df_by_hour['NO_rolling_std']= df_by_hour.NO.rolling(window=84).std()
df_by_hour['NO2_rolling_std']= df_by_hour.NO2.rolling(window=84).std()
df_by_hour['CO_rolling_std']= df_by_hour.CO.rolling(window=96).std()
df_by_hour['O3_rolling_std']= df_by_hour.O3.rolling(window=240).std()
# -

df_by_hour.isnull().sum()

# we are now dropping the first rows with null value which is added after feature added
df_by_hour =df_by_hour.dropna()
# For filling with median
#df_by_hour = df_by_hour.fillna(df.median())
# for filling with zeros
#df_by_hour.fillna(0.00, index =True)
df_by_hour.head(2)

#printing columns name
df_by_hour.columns


# ### Feature selection

# +
# Reference by Krish Naik- feature selection by correlation matrix
# Selection of feature based on Pearson's correlation analysis
# We will remove the highly correlated features (corelation among independent features)

def correlation_heatmap(df):
    
    plt.figure(figsize=(14,12))
    cor = df.drop(['Day_of_month', 'Month', 'Day'], axis=1).corr()
    svm =sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r) # family='Arial', fontsize=14
    figure= svm.get_figure()

    my_path = SAVING_DIR
    my_file = 'heatmap_correlation_matrix.png'
    figure.savefig(os.path.join(my_path, my_file))
    #figure.savefig('heatmap_correlation_matrix.png')
    #plt.show()


correlation_heatmap(df_by_hour)


# +
# let's wirte a function for selecting highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set() # set of all names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in abosolute coeff value
                colname= corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr


# -

corr_features = correlation(df_by_hour, 0.8)
print(len(set(corr_features)))
print(corr_features)

df_by_hour = df_by_hour.reset_index(drop=True)

# +
# Saving the data
#df_by_hour.to_csv('C:/Users/Rabbil/Documents/BDA_thesis/Notebooks/cleaned_df_final.csv')

DIR_TO_SAVE = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed'
csv_file_name ='cleaned_df_final.csv'
csv_file_loc = os.path.join(DIR_TO_SAVE, csv_file_name)

# df to csv file
df_by_hour.to_csv(csv_file_loc, index = False)
# -

# data = pd.read_csv(r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed\cleaned_df_final.csv').drop(['index'], axis=1)


