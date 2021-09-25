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
#plt.rcParams["axes.labelsize"] = 16

params = {'legend.fontsize': 17,
          'legend.handlelength': 2,
         'axes.labelsize': 'large'}
plt.rcParams.update(params)

from datetime import datetime # converting the timestamp data into datetime object

# +
# loading the saved file

SAVING_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\outputs\Vizualization'

DATA_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed\merged_data.csv'
data = pd.read_csv(DATA_DIR) 
data.head(5)


# -

def data_info(df):
    print('Numver of Rows :', df.shape[0])
    print('Number of Columns :', df.shape[1])
    print('\nFeatures : \n', df.columns.tolist())
    print('\nMissing values counts :\n', df.isnull().sum())
    print('\nUnique values counts : \n', df.nunique())
    print('percentage of null value : \n',df.isnull().sum()/len(df))
    cols=['NO', 'NO2', 'O3', 'CO', 'pressure', 'humidity', 'temp', 'NO_s', 'NO2_s', 'O3_s', 'CO_s']
    print('percentage of negative value: \n', df[df[cols]<0].count()/len(df))



basic_insight = data_info(data)
print(basic_insight)


# +
# Printing summary statistics

def summary_statistics(df):

    desc = df.describe()
    desc.loc['count'] = desc.loc['count'].astype(int).astype(str)
    desc.iloc[1:] = desc.iloc[1:].applymap('{:.2f}'.format)
    desc = desc.append(df.reindex(desc.columns, axis=1).agg(['skew', 'kurt']))
    
    # saving the statistics
    my_path = SAVING_DIR
    my_file = 'summary_statistics.xlsx'
    desc.to_excel(os.path.join(my_path, my_file))
    
    return desc


# +
#pip install openpyxl
# -

summary = summary_statistics(data)
summary


# +
# function for plotting the features/variables level

def plot_pollutant_concentrations(df):
    
    cols_plot = ['NO', 'NO2', 'O3', 'CO']
    df.set_index('date')[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', subplots=True, legend=True, figsize=(12,8))
    
    my_path = SAVING_DIR
    my_file = 'raw_pollutant_concentration_level.png'
    plt.savefig(os.path.join(my_path, my_file))
    

def plot_metorological_concentrations(df):
    
    cols_plot = ['pressure', 'humidity', 'temp']
    df.set_index('date')[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', subplots=True, legend=True, figsize=(12,8))
    
    my_path = SAVING_DIR
    my_file = 'raw_metorolgical_concentration_level.png'
    plt.savefig(os.path.join(my_path, my_file))
    


# -

plot_pollutant_concentrations(data)

plot_metorological_concentrations(data)


