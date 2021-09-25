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

# printing current working directory
cwd = os.getcwd() 
print(cwd)

# +
cwd = os.getcwd() # dirname (abspath(__file__))
BASE_DIR = os.path.dirname(cwd)

# We have two dataset
DATA_DIR = os.path.join(BASE_DIR, 'raw\Data_supersite')
DATA_DIR_1 = os.path.join(BASE_DIR, 'raw\Data_movingsite')
# -

# check what are the data inside
print(os.listdir(DATA_DIR))
print(os.listdir(DATA_DIR_1))


# +
# Renaming the csv file for supersite data
path = os.chdir(r"C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\raw\Data_supersite")

i = 2
for file in os.listdir(path):
    # use format function {}.format(value)
    new_file_name = "Month{}.csv".format(i)
    os.rename(file, new_file_name)
    
    i = i+1
# -

# list of files in the directory
for filename in os.listdir(DATA_DIR):
    print(filename)

# ### Combining all the csv files into one csv file

# +
cwd = os.getcwd() # dirname (abspath(__file__))
BASE_DIR = os.path.dirname(cwd)
DATA_DIR = os.path.join(BASE_DIR, 'Data_supersite')

# saving the combined data into this directory -creating a directory named cache_supersite
CACHE_DIR = os.path.join(BASE_DIR, 'cache_supersite')
os.makedirs(CACHE_DIR, exist_ok = True)

# creating another cache_dir for moving site data
CACHE_DIR_1 = os.path.join(BASE_DIR, 'cache_movingsite')
os.makedirs(CACHE_DIR_1, exist_ok = True)
# -

# reding only one file
df = pd.read_csv(os.path.join(DATA_DIR, 'Month2.csv'))
df.head()

# +
# combining all the csv files

# we have list of dataframes and take this list as
my_dataframes = [] # then append this with this_df

csv_files = [x for x in os.listdir(DATA_DIR) if x.endswith(".csv")] # to make sure my filenames are only filenames that have a CSV file
#for filename in os.listdir(DATA_DIR): # replace by csv_files

for filename in csv_files:
    print(filename)
    month = filename.replace(".csv", "")
    csv_path = os.path.join(DATA_DIR, filename)
    this_df = pd.read_csv(csv_path)
    
    # now adjust this_df
    this_df['filename']= filename
    this_df['Month'] = month
    print(this_df.head(n=1))
    my_dataframes.append(this_df)

# +
# my_entire_dataframe is pd.DataFrame(my_dataframes) 
# creating data frame using another dataframe

# now combine dataframe using concat method
my_entire_dataframe = pd.concat(my_dataframes)
my_entire_dataframe.head(5)

# +
# Saving entire dataframe into a new csv file
# Exporting the data to one single source (cache source)

dataset = os.path.join(CACHE_DIR, 'super-site-dataset.csv') # we will save it in CACHE_DIR
my_entire_dataframe.to_csv(dataset, index=False)

# +
# combining all the csv files for moving site

# we have list of dataframes and take this list as
my_dataframes = [] # then append this with this_df

csv_files = [x for x in os.listdir(DATA_DIR_1) if x.endswith(".csv")] # to make sure my filenames are only filenames that have a CSV file
#for filename in os.listdir(DATA_DIR): # replace by csv_files

for filename in csv_files:
    print(filename)
    month = filename.replace(".csv", "")
    csv_path = os.path.join(DATA_DIR_1, filename)
    this_df = pd.read_csv(csv_path)
    
    # now adjust this_df
    this_df['filename']= filename
    this_df['Month'] = month
    print(this_df.head(n=1))
    my_dataframes.append(this_df)
# -

# now combine dataframe using concat method
my_entire_dataframe_ms = pd.concat(my_dataframes)
my_entire_dataframe_ms.tail(5)

# +
# Saving entire dataframe into a new csv file
# Exporting the data to one single source (cache source)

dataset_ms = os.path.join(CACHE_DIR_1, 'moving-site-dataset.csv') # we will save it in CACHE_DIR_1
my_entire_dataframe_ms.to_csv(dataset_ms, index=False)

# +
# Data pipelining using OOPs concept

import os
import pandas as pd

cwd = os.getcwd() # dirname (abspath(__file__))
#print(cwd)
BASE_DIR = os.path.dirname(cwd)

# We have two dataset
DATA_DIR_SS = os.path.join(BASE_DIR, 'raw\Data_supersite')
DATA_DIR_MS = os.path.join(BASE_DIR, 'raw\Data_movingsite')

# check what are the data inside
#print(os.listdir(DATA_DIR))
#print(os.listdir(DATA_DIR_1))

# saving the combined data into this directory -creating a directory named cache_supersite
CACHE_DIR_SS = os.path.join(BASE_DIR, 'cache_supersite')
os.makedirs(CACHE_DIR_SS, exist_ok = True)

# creating another cache_dir for moving site data
CACHE_DIR_MS = os.path.join(BASE_DIR, 'cache_movingsite')
os.makedirs(CACHE_DIR_MS, exist_ok = True)


# Renaming the csv file
#path/DATA_DIR = os.chdir(r"C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\raw\Data_supersite")

def rename_file(data_dir):
    '''
    Renaming the csv files in dataset; take data_dir or path as argument
    Also print the list of renamed files
    '''
    i = 2
    for file in os.listdir(data_dir):
        # use format function {}.format(value)
        new_file_name = "Month{}.csv".format(i)
        os.rename(file, new_file_name)

        i = i+1
        
    # check the list of files in the directory
    for filename in os.listdir(data_dir):
        print(filename)




# Function for combining sevearl files and saving into one file

def combine_and_save_data(data_dir, cache_dir):
    '''
    In the function two arguments are passed, the data_dir whcih is source of data
    Argument cache_dir is the source where the combined data will be saved
    '''
    # declare the empty list of dataframes 
    my_dataframes = [] 

    csv_files = [x for x in os.listdir(data_dir) if x.endswith(".csv")] # to make sure my filenames are only filenames that have a CSV file
    #for filename in os.listdir(DATA_DIR): # replace by csv_files

    for filename in csv_files:
        print(filename)
        month = filename.replace(".csv", "")
        csv_path = os.path.join(data_dir, filename)
        this_df = pd.read_csv(csv_path)

        # now adjust this_df
        this_df['filename']= filename
        this_df['Month'] = month
        
        # printing the number of dataframe 
        print(this_df.head(n=1))
        
        # append earlier dataframe with later dataframe
        my_dataframes.append(this_df)
        
    # now combine dataframe using concat method
    my_entire_dataframe = pd.concat(my_dataframes)
        
    # Saving entire dataframe into a new csv file
    # Exporting the data to one single source (cache source)
    dataset = os.path.join(cache_dir, 'moving-site-dataset1.csv') 
    
    # we will save it in CACHE_DIR
    my_entire_dataframe.to_csv(dataset, index=False)
    
    return dataset

    
# -

combine_and_save_data(DATA_DIR_1, CACHE_DIR_1)


# +
# Renaming the csv file for supersite data
#path/DATA_DIR = os.chdir(r"C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\raw\Data_supersite")

def rename_file(path):
    i = 2
    for file in os.listdir(path):
        # use format function {}.format(value)
        new_file_name = "Month{}.csv".format(i)
        os.rename(file, new_file_name)

        i = i+1
        
    # list of files in the directory
    for filename in os.listdir(path):
        print(filename)


# -

rename_file(DATA_DIR)


