# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/ingest//py
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

import os
import pandas as pd
import numpy as np
from datetime import datetime # converting the timestamp data into datetime object

# +
# moving site data reading
DATA_PATH = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\raw\cache_movingsite\moving-site-dataset.csv'

df_moving_site = pd.read_csv(DATA_PATH)
df_moving_site.head(5)

# +
# super site data reading
DATA_PATH_1 = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\raw\cache_supersite\super-site-dataset.csv'

df_super_site = pd.read_csv(DATA_PATH_1)
df_super_site.head(5)


# +
# function for basic cleaning of the dataset

def clean_data(df):
    
    # renaming columns
    df = df.rename(columns = {'timestamp': 'date','BME680_pressure':'pressure', 'BME680_humidity':'humidity', 'BME680_temperature':'temp'})
    # we see that date column is integer, so need to work on it
    # converting the timestamp data (integer column) into datetime object 
    df['date'] = pd.to_datetime(df['date'],unit='s')
    # droping the unnecessary features
    df.drop(['comment', 'filename', 'Month'], axis=1, inplace = True)
    
    return df



# +
df_moving_site = clean_data(df_moving_site)
df_super_site = clean_data(df_super_site)

# adding suffix to each columns
df_super_site = df_super_site.add_suffix('_s')
print(df_moving_site.shape, df_super_site.shape)
# -

# nerging two dataframe on 'date' column
merged_df = pd.concat([df_moving_site, df_super_site], axis=1)
merged_df.columns

# creating new dataframe keeping some columns and droping others
merged_df= merged_df.iloc[:, np.r_[0:11,15:19]]
merged_df.head(5)

# +
# Save the file
SAVING_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed'
csv_file_name ='merged_data.csv'
csv_file_loc = os.path.join(SAVING_DIR, csv_file_name)

# df to csv file
merged_df.to_csv(csv_file_loc, index=False) # here index is used to avoid unwanted index-like column named unnamed:0 during reading csv file
# -


