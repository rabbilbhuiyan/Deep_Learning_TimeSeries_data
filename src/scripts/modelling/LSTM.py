# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/modelling//py
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

# ## Long Short-Term Memory (LSTM) neural network model

# +
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os
SAVING_DIR = r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\outputs\models'

import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import sklearn.metrics as metrics # for model evaluation 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -

df_by_hour = pd.read_csv(r'C:\Users\Rabbil\Documents\BDA_thesis\thesis-project\data\processed\cleaned_df_final.csv').drop(['index'], axis=1)
df_by_hour.head(2)

df_by_hour.set_index('date', inplace=True)
df_by_hour.head(2)

# sub-selecting data 
df_by_hour= df_by_hour.loc['2019-03':'2019-06']
df_by_hour.drop(['Day'], axis=1, inplace =True)

df_by_hour.columns

# ### Machine learning data preparation 

# +
# Spliting the data into train and test sets
train_size = int(len(df_by_hour)*0.8)
test_size = len(df_by_hour) - train_size

train, test = df_by_hour.iloc[0:train_size], df_by_hour.iloc[train_size:len(df_by_hour)]

print(train.shape, test.shape)

# +
# Scaling of data

# creating variables for list of columns with scaler
feature_columns = ['Pressure', 'Humidity', 'Temperature','NO', 'NO2', 'CO',
         'NO_s', 'NO2_s','O3_s', 'CO_s', 'Hour','Day_of_week', 'is_weekday', 'daypart',
        'NO_rolling_mean', 'NO2_rolling_mean', 'CO_rolling_mean',
        'O3_rolling_mean', 'NO_rolling_min', 'NO2_rolling_min',
        'CO_rolling_min', 'O3_rolling_min', 'NO_rolling_max', 'NO2_rolling_max',
        'CO_rolling_max', 'O3_rolling_max',
        'O3_rolling_std']
#target_columns = ['NO']

feature_transformer = MinMaxScaler()
NO_transformer = MinMaxScaler()

# fit the scaler on training data
feature_transformer = feature_transformer.fit(train[feature_columns])
NO_transformer = NO_transformer.fit(train[['NO']])

# +
train.loc[:, feature_columns] = feature_transformer.transform(train[feature_columns])
train['NO'] = NO_transformer.transform(train[['NO']])

test.loc[:, feature_columns] = feature_transformer.transform(test[feature_columns])
test['NO'] = NO_transformer.transform(test[['NO']])

# +
# Making the time series data into sub sequences by writing a funciton
# split a multivariate sequene into samples
import numpy as np 
def create_dataset(X, y, time_steps=1): # X=features, y=labels, time_steps= history of sequence
    Xs, ys = [], [] # create two lists
    for i in range(len(X) - time_steps): #loops for the subsequent steps
        values = X.iloc[i: (i + time_steps)]
        Xs.append(values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

#x = np.asarray(x).astype('float32')
#x_train = np.asarray(x_train).astype(np.float32)
#y_train = np.asarray(y_train).astype(np.float32)


# +
# Specify number of time steps
TIME_STEPS = 48 # history of 24 hours to predict for the next 24 hours

# Creating actual taining and test dataset
# Reshaping to samples, time_steps and n_features
X_train, y_train = create_dataset(train, train.NO, time_steps= TIME_STEPS)
X_test, y_test = create_dataset(test, test.NO, time_steps= TIME_STEPS)
# -

# printing number of samples, number of time_steps and number of features
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# print the first sequence
#X_test[0].shape
X_test[0][0].shape, X_train[0][0].shape


# +
# Creating function for deep learning model

def create_model(optimizer='adam', dropout_rate=0.2, activation='relu'):
    model_lstm = keras.Sequential()

    model_lstm.add( keras.layers.LSTM(units = 128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_lstm.add(keras.layers.Dropout(rate=0.2))# add dropout layer

    model_lstm.add(keras.layers.Dense(units=1))

    # compile the model
    model_lstm.compile(loss='mean_squared_error', optimizer='RMSProp')
    
    return model_lstm


model_lstm = create_model()
model_lstm.summary()

# Saving the model
my_path = SAVING_DIR
my_file = "model_lstm.h5"
model_lstm.save_weights(os.path.join(my_path, my_file))


# +
# %%time
seed = 7 # fix random seed for reproducibility
np.random.seed(seed)

lstm_history = model_lstm.fit(X_train, y_train, 
                        epochs = 30,
                        batch_size=32,
                        validation_split=0.3,
                        shuffle=False
                       )

# +
# Plot the loss function
fig, ax = plt.subplots(1, 1, figsize=(10,6))
plt.plot(lstm_history.history['loss'], 'r', label='train')
plt.plot(lstm_history.history['val_loss'], 'b', label='validation')
ax.set_xlabel(r'Epoch', fontsize=16)
ax.set_ylabel(r'Loss', fontsize=16)
ax.legend(fontsize = 16)
ax.tick_params(labelsize=16)


# Save the plot
my_path = SAVING_DIR
my_file = 'LSTM_model_NO.png'
plt.savefig(os.path.join(my_path, my_file))
# -

# making prediction
y_pred_lstm = model_lstm.predict(X_test)


# +
# function for calculating MAPE
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual))*100

# Model evaluation using mean square error (MSE)
print('r Squared Error:' , metrics.r2_score(y_test, y_pred_lstm))
print('Mean Squared Error:' , metrics.mean_squared_error(y_test, y_pred_lstm))
# Model evaluation using  root mean squared error (RMSE): value 1 deptics worst model
print('Root Mean Squared Error:' , np.sqrt(metrics.mean_squared_error(y_test, y_pred_lstm)))
# Model evaluation using mean absolute error (MAE): value 0 deptics best model
print('Mean Absolute Error:' , metrics.mean_absolute_error(y_test, y_pred_lstm))
print('MAPE:' , mape(y_test, y_pred_lstm))

# +
# use inverse the scaler for the feature variable
# invert the X_train, y_train and predicted data
y_train_inv = NO_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = NO_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = NO_transformer.inverse_transform(y_pred_lstm)

# plotting the predicted value
fig, ax = plt.subplots(1, 1, figsize=(10,6))
plt.plot(y_test_inv.flatten(), marker='.', label='True')
plt.plot(y_pred_inv.flatten(), 'r', label='Predicted')

ax.set_xlabel(r'Timesteps (hour)', fontsize=16)
ax.set_ylabel(r'NO (ppm)', fontsize=16)
ax.legend()
ax.tick_params(labelsize=16)
#plt.ylabel('Global_active_power', size=15)
#plt.xlabel('Time step', size=15)
plt.legend(fontsize=18)

my_path = SAVING_DIR
my_file = 'LSTM_prediction_NO'
plt.savefig(os.path.join(my_path, my_file), dpi=300, bbox_inches='tight')
# -

# ### Tunning hyperparameter using cross vlaidation : by using GridSearchCV from Scikit-Learn

# +
# %%time
seed = 7 # fix random seed for reproducibility
np.random.seed(seed)


# create the sklearn model for the network
model_batch_epoch = KerasRegressor(build_fn=create_model, verbose=1)

# we choose the initializers that came at the top in our previous cross-validation!!
#init_mode = ['glorot_uniform', 'uniform'] 
#layer_size= [32, 64, 128]
batches = [16, 32, 64]
epochs = [20, 30, 40]
#learn_rate= [0.001,0.01, 0.1 ]
dropout_rate=[0.2, 0.3]
optimizer=['Adam', 'RMSProp']
#activation=['sigmoid', 'relu']

# grid search for initializer
param_grid = dict( batch_size=batches, epochs=epochs, dropout_rate=dropout_rate, optimizer=optimizer)
grid = GridSearchCV(estimator=model_batch_epoch,
                    param_grid=param_grid,
                    cv=3)
grid_result = grid.fit(X_train, y_train)
# -

# print the results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')

# function for calculating MAPE
y_pred_lstm_grid = grid.predict(X_test)

# +
# Model evaluation using different metrics

print('r Squared Error:' , metrics.r2_score(y_test, y_pred_lstm_grid))
print('Mean squred error:', metrics.mean_squared_error(y_test, y_pred_lstm_grid))
# Model evaluation using  root mean squared error (RMSE): value 1 deptics worst model
print('Root Mean Squared Error:' , np.sqrt(metrics.mean_squared_error(y_test, y_pred_lstm_grid)))
# Model evaluation using mean absolute error (MAE): value 0 deptics best model
print('Mean Absolute Error:' , metrics.mean_absolute_error(y_test, y_pred_lstm_grid))
print('Mean absolute percentage error:', mape(y_test, y_pred_lstm_grid))
# -


