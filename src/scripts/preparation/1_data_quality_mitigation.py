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
# -

SAVING_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\outputs\Vizualization'

# loading the saved file
DATA_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed\merged_data.csv'
df = pd.read_csv(DATA_DIR) 
df.head(2)

# Deleting the highest peak e.g 1000 ppm for CO which was cuased due to power on the tram
df.drop(df[df.CO > 50].index, inplace=True)

# +
# Applying background compensation to the whole dataset

# df has a RangeIndex, so we get to slice 
group_size = 3*60
# data frame has been sorted before this point and the rows are in the correct order
slices = df[::group_size]

# but you don't want the group number to be the ordinal at the slices
# so make a copy of the slice to assign good group numbers to it (or get a chained assignment warning)
slices=slices.copy()
slices['group'] = [i for i in range(len(slices))]
df['group'] = slices['group']

# ffill with the nice group numbers
df['group'].ffill(inplace=True)

#now trim the last group
last_group = df['group'].max()
if len(df[df['group']==last_group]) < group_size:
    df = df[df['group'] != last_group]
print(df)
# -

print(df.group.nunique())

# set index of date column
# making dataframe's index as DatetimeIndex, also group column
df.set_index(['date', 'group'], inplace=True)

df.index[5]

# applying formula for background compensation of the data
back_g = lambda x: x-x.median()
df['NO_b']=df[['NO']].groupby(level='group').transform(back_g)
df['NO2_b']=df[['NO2']].groupby(level='group').transform(back_g)
df['O3_b']=df[['O3']].groupby(level='group').transform(back_g)
df['CO_b']=df[['CO']].groupby(level='group').transform(back_g)
df.head(5)


# +
# plotting the variables after background compensations

def plot_pollutant_concentrations_b(df):
    
    cols_plot_b = ['NO_b', 'NO2_b', 'O3_b', 'CO_b']
    df.reset_index('group')[cols_plot_b].plot(marker='.', alpha=0.5, linestyle='None', subplots=True, legend=True, figsize=(12,8))
    
    my_path = SAVING_DIR
    my_file = 'background_compensated_pollutant_concentration_level.png'
    plt.savefig(os.path.join(my_path, my_file))
    
plot_pollutant_concentrations_b(df)
# -

# dropping the raw variables (before background compensated)
df = df.drop(['NO', 'NO2', 'O3', 'CO'], axis=1)

# +
# handling negative value using Polissar method
neg_v = lambda x: x[x>0].mean()/2
df['NO_p']=df[['NO_b']].groupby(level='group').transform(neg_v)
df['NO2_p']=df[['NO2_b']].groupby(level='group').transform(neg_v)
df['O3_p']=df[['O3_b']].groupby(level='group').transform(neg_v)
df['CO_p']=df[['CO_b']].groupby(level='group').transform(neg_v)

df['NO_s']=df[['NO_s']].groupby(level='group').transform(neg_v)
df['NO2_s']=df[['NO2_s']].groupby(level='group').transform(neg_v)
df['O3_s']=df[['O3_s']].groupby(level='group').transform(neg_v)
df['CO_s']=df[['CO_s']].groupby(level='group').transform(neg_v)


df.head(3)

# +
# drop the varialves before negative value handling
df= df.drop(['NO_b', 'NO2_b', 'CO_b', 'O3_b'], axis=1)

print('percentage of negative value: \n', df[df<0].count()/len(df))
# -

# renaming the column after all preprocessing
df.rename(columns={'NO_p':'NO', 'NO2_p':'NO2', 'O3_p':'O3', 'CO_p':'CO','pressure':'Pressure', 'humidity':'Humidity', 'temp':'Temperature'}, inplace=True)


# +
# plotting the variables after all processing

def pollutant_concentration_final_plot(df):
    cols_plot = ['NO', 'NO2', 'O3', 'CO']
    axes = df.reset_index('group')[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(14,9), subplots=True)
    for ax in axes:
        ax.set_ylabel('ppm')
        ax.set_xlabel('')
        
    my_path = SAVING_DIR
    my_file = 'pollutant_concentration_level_final.png'
    plt.savefig(os.path.join(my_path, my_file))
    
pollutant_concentration_final_plot(df)


# +
def pollutant_concentration_final_plot_spatial(df):
    cols_plot = ['NO_s', 'NO2_s', 'O3_s', 'CO_s']
    axes = df.reset_index('group')[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(14,9), subplots=True)
    for ax in axes:
        ax.set_ylabel('ppm')
        ax.set_xlabel('')
        
    my_path = SAVING_DIR
    my_file = 'pollutant_concentration_level_final_spatial_data.png'
    plt.savefig(os.path.join(my_path, my_file))
    
pollutant_concentration_final_plot_spatial(df)

# +
# Save the file after background compensation and negative value handling

DIR_TO_SAVE = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed'
csv_file_name ='background_negative_value_processed_data.csv'
csv_file_loc = os.path.join(DIR_TO_SAVE, csv_file_name)

# df to csv file
df.to_csv(csv_file_loc) # here index is used to avoid unwanted index-like column named unnamed:0 during reading csv file
# -


