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
sns.set(rc={'figure.figsize':(11,7)}, style="white", font_scale=1.4)#style="whitegrid"
plt.rcParams["axes.labelsize"] = 16

from datetime import datetime # converting the timestamp data into datetime object
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # determining autocorrelation
from scipy import stats # detecting outliers by z score

# +
SAVING_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\outputs\Vizualization'

DATA_PATH = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed\background_negative_value_processed_data.csv'
df = pd.read_csv(DATA_PATH)
df.head(4)
# -

# converting the timestamp data into datetime object 
df['date'] = pd.to_datetime(df['date'])
df.date.dtype

# set index of date and group column
df.set_index(['date', 'group'], inplace=True)

# Checking the null value
df.isnull().sum()

# forward filling
#Tuesday data (missing) equals to Monday data (existing) is forward filling. The opposite is backward filling
df.fillna(method='ffill', inplace=True)
df.isnull().sum()


# +
# plotting the variables after all processing

def pollutant_concentration_final_plot_ffill(df):
    cols_plot = ['NO', 'NO2', 'O3', 'CO']
    axes = df.reset_index('group')[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(14,9), subplots=True)
    for ax in axes:
        ax.set_ylabel('concentration %')
        ax.set_xlabel('')
        
    my_path = SAVING_DIR
    my_file = 'pollutant_concentration_level_final_ffill.png'
    plt.savefig(os.path.join(my_path, my_file))
    
pollutant_concentration_final_plot_ffill(df)


# +
#finding outliers by running the summary statistics on the variables.

def preprocessed_summary_statistics(df):

    desc = df.describe()
    desc.loc['count'] = desc.loc['count'].astype(int).astype(str)
    desc.iloc[1:] = desc.iloc[1:].applymap('{:.2f}'.format)
    desc = desc.append(df.reindex(desc.columns, axis=1).agg(['skew', 'kurt']))
    
    # saving the statistics
    my_path = SAVING_DIR
    my_file = 'preprocessed_summary_statistics.xlsx'
    desc.to_excel(os.path.join(my_path, my_file))
    
    return desc


# -

summary = preprocessed_summary_statistics(df)
summary


# +
# the skewness value should be between -1 and +1, 
# and any major deviation from this range indicates the presence of extreme values.

# +
# dist plot and box plot together
# Cut the window in 2 parts

def dist_box_plot_together(df):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    # Add a graph in each part
    #sns.boxplot(df["sepal_length"], ax=ax_box)
    svm=sns.boxplot(df.reset_index('group').resample('H').mean().O3, ax=ax_box)
    svm=sns.distplot(df.reset_index('group').resample('H').mean().O3, ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

    figure= svm.get_figure()
    
    my_path = SAVING_DIR
    my_file = 'box_dist_plot_O3.png'
    figure.savefig(os.path.join(my_path, my_file))
    #figure.savefig('boxplot_distplot_NO2.png')


# -

dist_box_plot_together(df)


# +
# Replacing outlier with median
def outlier_replace(df):
    df['CO'] = np.where(df['CO'] > df['CO'].quantile(0.95), df['CO'].quantile(0.50), df['CO']) # replaces all values, which are > the 95th percentile, with the median value
    df['NO'] = np.where(df['NO'] > df['NO'].quantile(0.95), df['NO'].quantile(0.50), df['NO'])
    df['NO2'] = np.where(df['NO2'] > df['NO2'].quantile(0.95), df['NO2'].quantile(0.50), df['NO2'])
    df['O3'] = np.where(df['O3'] > df['O3'].quantile(0.95), df['O3'].quantile(0.95), df['O3'])
    df['Humidity'] = np.where(df['Humidity'] > df['Humidity'].quantile(0.95), df['Humidity'].quantile(0.50), df['Humidity'])
    df['Temperature'] = np.where(df['Temperature'] > df['Temperature'].quantile(0.95), df['Temperature'].quantile(0.50), df['Temperature'])
    
    
outlier_replace(df) 


# -

# vizualization of the data after outlier replacement with median and performing resmaple
def pollutant_plot_outlier_replace(df):
    ax =df.reset_index('group').NO.resample('H').mean().fillna(df.NO.median()).plot(title='NO resampled & outlier replc', color='orange') 
    plt.tight_layout()
    
        
    my_path = SAVING_DIR
    my_file = 'NO_after_outlier_replace.png'
    plt.savefig(os.path.join(my_path, my_file))


pollutant_plot_outlier_replace(df)

# +
# Based on these plots above, it is clear that data upto June 30 would be good for prediction plot as there is few spikes
# -

df =df.reset_index(['date', 'group'])
df.head(2)

df.shape

# +
# saving the data at this stage (before feature engineering)- this data can be used for geospatial plotting

DIR_TO_SAVE = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed'
csv_file_name ='outlier_replace_processed_data.csv'
csv_file_loc = os.path.join(DIR_TO_SAVE, csv_file_name)

# df to csv file
df.to_csv(csv_file_loc, index=False) # here index is used to avoid unwanted index-like column named unnamed:0 during reading csv file
# -


