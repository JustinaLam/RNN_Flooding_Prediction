# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# Requires all tensorflow dependencies
try:
  import tensorflow.keras as keras 
except:
  print("Error: Tensorflow import failed")
  exit(0)

# import datetime
from datetime import *
import math
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


fig_test = Figure()
fig_valid = Figure()

figs = [] #fig_test, fig_valid
results = [] # datetime, yhat_valid, y_validation

toReturn = []

def init(scope, month, day, year):
  makePredictions(scope, int(month), int(day), int(year))
  print(len(figs))
  print(len(toReturn))
  return toReturn

# Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = DataFrame(data)
  cols, names = list(), list()
  # Input sequence 
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # Forecast sequence 
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # Combine
  agg = concat(cols, axis=1)
  agg.columns = names
  # Drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  agg
  return agg

def makePredictions(scope, month, day, year): 
  # Load dataset
  dataset = pd.read_csv("FannoCreekData6950_2000-2021_To_CSV.csv")
  # Drop unnecessary column
  dataset.drop(dataset[dataset.agency_cd != "USGS"].index, inplace=True)

  cols = [c for c in dataset.columns if not (c.endswith('cd'))]
  dataset = dataset[cols]


  # Site_no to numeric
  dataset['site_no'] = pd.to_numeric(dataset['site_no'])
  # Datetime to datetime
  dataset['datetime'] = pd.to_datetime(dataset['datetime'],errors='coerce')
  # Gage_height to float
  dataset['Gage_height,feet'] = pd.to_numeric(dataset['Gage_height,feet'])
  dataset.rename(columns = {'Gage_height,feet':'Gage_height'}, inplace = True)
  # Discharge to float
  dataset['Discharge,cubic_feet_per_second'] = pd.to_numeric(dataset['Discharge,cubic_feet_per_second'])
  dataset.rename(columns = {'Discharge,cubic_feet_per_second':'Discharge'}, inplace = True)
  # Water_temp
  dataset.rename(columns = {'Temperature_water_C':'Water_temp'}, inplace = True)
  dataset['Water_temp'] = pd.to_numeric(dataset['Water_temp'])
  # Conductance
  dataset.rename(columns = {'Specific_conductance_water_uScm':'Conductance'}, inplace = True)
  dataset['Conductance'] = pd.to_numeric(dataset['Conductance'])
  # Dissolved_oxygen
  dataset.rename(columns = {'Dissolved_oxygen_water_mgL':'Dissolved_oxygen'}, inplace = True)
  dataset['Dissolved_oxygen'] = pd.to_numeric(dataset['Dissolved_oxygen'])
  # Turbidity
  dataset.rename(columns = {'Turbidity_water_FNU':'Turbidity'}, inplace = True)
  dataset['Turbidity'] = pd.to_numeric(dataset['Turbidity'])


  # Set flood stages and add resulting flood status (Y/N) column 
  dataset["fld_stg"] = np.where(dataset['site_no'] == 14206950, 9.1, 11.1)
  dataset["fld_YN"] = np.where(dataset['Gage_height'] >= dataset['fld_stg'], 1, 0)


  # Add column for Gage_height +7 days into future (repeat for +14, +21, +28)
  # Original dataset has 96 rows of data per day
  shiftN = 30 if (scope == "30day") else (365 if (scope == "year") else 7)
  df_validation = dataset.tail(shiftN * 96).copy()

  dataset["Gage_height_shift"] = (dataset.copy())["Gage_height"]
  dataset["Gage_height_shift"] = dataset.Gage_height_shift.shift(-shiftN * 96)

  last_date = pd.to_datetime(dataset['datetime'].dt.date.iloc[-1], errors='coerce')

  # ------------------------------- TRIM DATASET -------------------------------

  # Takes into account user's choice of form input (stored in scope variable)
  # This step is necessary in order to ensure that the flooding-positive events carry sufficient weight
  # to influence the model despite being rare in the dataset, relative to the number of flooding-negative rows

  # Training dataset must be sufficiently balanced between flooding-positive and flooding-negative events
  # so that the model cannot achieve high accuracy by simply making only negative predictions.

  # This is further prevented by basing the model's predictions on the continuous variable gage_height, which
  # measures water level, instead of only the discrete categories of "Yes" and "No" for flooding.

  if (scope == "7day_daily"):
    # Every day plus all flooding-positive rows *************

    datasetOrig = dataset.copy(deep=True)
    dataset = dataset[((dataset.datetime.dt.hour == 0) & (dataset.datetime.dt.minute == 0)) | (dataset.fld_YN == 1)]

    # Set Gage_height of each day to the maximum value for that day, since dataset contains hourly data but
    # form input indicates that only daily values should be considered
    dataset['Gage_height'] = datasetOrig['Gage_height'].rolling(96).max().shift(-95)


# -------------------------------

  else:

    if (scope == "7day_6hrs"):
      
    # Every 6 hours plus all flooding-positive rows *************
      dataset = dataset[(dataset.datetime.dt.hour % 6 == 0) | (dataset.fld_YN == 1)]
      dataset = dataset[(dataset.datetime.dt.minute == 0) | (dataset.fld_YN == 1)]

    # -------------------------------

    else:

      # Every 12 hours plus all flooding-positive rows (default) ************* 

      dataset = dataset[(dataset.datetime.dt.hour % 12 == 0) | (dataset.fld_YN == 1)]
      dataset = dataset[(dataset.datetime.dt.minute == 0) | (dataset.fld_YN == 1)]


    # ----------------------------------------------------------------------------

  # Count YN values to check for balance between flooding-positive and flooding-negative events
  print(len(dataset[dataset['fld_YN'] == 1])) # Flooding-positive
  print(len(dataset[dataset['fld_YN'] == 0])) # Flooding-negative


  # Drop unnecessary columns
  dataset = dataset.drop('site_no',1, errors='ignore')
  dataset = dataset.drop([c for c in dataset if c.endswith('_cd')], 1, errors='ignore')

  # Drop any negative values
  dataset.drop(dataset[dataset['Gage_height'] < 0].index, inplace=True)
  dataset.drop(dataset[dataset['Discharge'] < 0].index, inplace=True)


  # Replace all NaNs with value from previous row, the exception being Gage_height;
  # Only consider rows with valid Gage_height values
  dataset = dataset[dataset['Gage_height'].notna()]

  for col in dataset:
    dataset[col].fillna(method='pad', inplace=True)

  # Remove any NaNs or infinite values
  dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)] 


  # SUMMARIZE DATASET
  print(dataset.groupby('fld_YN').size()) # class distribution

  # -------------------------------------------------------------------------------
  # ---------------------------- COPY AND PASTE ENDS ------------------------------
  # -------------------------------------------------------------------------------

  # Move Gage_height to last column, as the value we are predicting
  dataset = dataset[['datetime','fld_stg'] + [c for c in dataset if c not in ['datetime','fld_stg','fld_YN', 'Gage_height_shift']] + ['fld_YN', 'Gage_height_shift']]

  # Predict continuous variable Gage_height_shift, instead of the discrete variable fld_YN, whose column is dropped
  dataset = dataset.drop('fld_YN',1) 

  # Create validation data -------------
  last_date = date(year,month,day)
  df_validation = dataset.copy()
  d1 = last_date
  d2 = last_date + timedelta(shiftN)
  df_validation = df_validation.drop(df_validation[df_validation['datetime'].dt.date < d1].index)

  df_validation = df_validation.drop(df_validation[df_validation['datetime'].dt.date > d2].index)

  print("VALIDATION BOUNDS:")
  print(df_validation.shape)
  print(df_validation.head(20))
  print(df_validation.tail(20))


  # Scale Data -------------

  # Data cleaning: Drop columns known to have little effect on predicted variable;
  # in this case, time and date labels, and repetitive values included as flood stage markers for reference
  df_relevant = dataset.copy()
  df_relevant = df_relevant.drop('datetime',1)
  df_relevant = df_relevant.drop('fld_stg',1)

  values = df_relevant.values
  print("Relevant Columns:")
  print(df_relevant.columns)

  # Specify columns to plot
  groups = [0, 1, 2, 3, 4, 5, 6, 7]
  i = 1
  # Plot each column
  pyplot.figure()
  for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
  pyplot.show()


  # ensure all data is float
  values = values.astype('float32')

  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)

  # frame as supervised learning
  reframed = series_to_supervised(scaled, 1, 1)


  # Repeat for validation data
  df_validation_relevant = df_validation.copy()
  df_validation_relevant = df_validation_relevant.drop('datetime',1)
  df_validation_relevant = df_validation_relevant.drop('fld_stg',1)
  validation_vals = df_validation_relevant.values
  validation_vals = validation_vals.astype('float32')
  validation_scaled = scaler.fit_transform(validation_vals)
  validation_reframed = series_to_supervised(validation_scaled, 1, 1)


  # Find and scale flood stages -------------

  fld_index = dataset.index[dataset['Gage_height'] == dataset['fld_stg']].tolist() # get indices of rows where Gage_height == fld_stg
  if (len(fld_index) == 0): # none exactly equal, but minimum difference
    fld_index = dataset.index[abs(dataset['Gage_height'] - dataset['fld_stg']) == min(abs(dataset['Gage_height'] - dataset['fld_stg']))].tolist()

  fld_index = fld_index[0]
  fld_stg_value = dataset.iloc[0, dataset.columns.get_loc('fld_stg')]
  print("FLOOD STAGE: " + str(fld_stg_value))

  min_gh = dataset['Gage_height'].min()
  max_gh = dataset['Gage_height'].max()

  df_fldstg = pd.DataFrame({'Gage_height':[min_gh,fld_stg_value,max_gh]})

  fld_stg_scaled = scaler.fit_transform(df_fldstg.values)
  fld_stg_scaled = fld_stg_scaled[1][0]

  fld_stgs_scaled = [fld_stg_scaled, (0.75 * fld_stg_scaled), (0.5 * fld_stg_scaled), (0.25 * fld_stg_scaled)]
  stg_colors = ['r','tab:orange','y','g']
  stg_labels = ['Flood Stage','75%','50%','25%']
  print("SCALED FLOOD STAGES:")
  for f in fld_stgs_scaled:
    print(str(f))



  df_validation.append(pd.Series(), ignore_index=True)
  df_validation.iloc[-1, df_validation.columns.get_loc('Gage_height')] = fld_stg_value

  fld_index_valid = df_validation.index[df_validation['Gage_height'] == df_validation['fld_stg']].tolist() # get indices of rows where Gage_height == fld_stg
  if (len(fld_index_valid) == 0): # none exactly equal, but minimum difference
    fld_index_valid = df_validation.index[abs(df_validation['Gage_height'] - df_validation['fld_stg']) == min(abs(df_validation['Gage_height'] - df_validation['fld_stg']))].tolist()

  fld_index_valid = fld_index_valid[0]
  fld_stg_valid_value = df_validation.iloc[0, df_validation.columns.get_loc('fld_stg')]

  print("VALIDATION MIN AND MAX")
  min_gh_valid = df_validation['Gage_height'].min()
  print(min_gh_valid)
  max_gh_valid = df_validation['Gage_height'].max()
  print(max_gh_valid)

  df_fldstg_valid = pd.DataFrame({'Gage_height':[min_gh_valid,fld_stg_valid_value,max_gh_valid]})

  fld_stg_valid_scaled = scaler.fit_transform(df_fldstg_valid.values)
  fld_stg_valid_scaled = fld_stg_valid_scaled[1][0]

  fld_stgs_valid_scaled = [fld_stg_valid_scaled, (0.75 * fld_stg_valid_scaled), (0.5 * fld_stg_valid_scaled), (0.25 * fld_stg_valid_scaled)]
  stg_colors = ['r','tab:orange','y','g']
  stg_labels = ['Flood Stage','75%','50%','25%']





  # Split into train and test sets
  values = reframed.values
  n_train_hours = math.floor(len(dataset.index) * 0.8)
  train = values[:n_train_hours, :]
  test = values[n_train_hours:, :]

  # Split into input and outputs
  train_X, train_y = train[:, :-1], train[:, -1]
  test_X, test_y = test[:, :-1], test[:, -1]

  # Reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

  # Repeat for validation data
  valid_vals = validation_reframed.values
  X_validation, y_validation = valid_vals[:, :-1], valid_vals[:, -1]
  X_validation = X_validation.reshape((X_validation.shape[0], 1, X_validation.shape[1]))

  # Design network
  model = Sequential()
  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dense(1,activation = keras.activations.sigmoid))
  model.compile(loss='mae', optimizer='rmsprop', metrics = ['mse','mae'])
  # Fit network
  history = model.fit(train_X, train_y, epochs=55, batch_size=100, validation_data=(test_X, test_y), verbose=2,  shuffle=False) #validation_split= 0.2)

  # Plot history
  pyplot.plot(history.history['loss'], label='train')
  pyplot.plot(history.history['val_loss'], label='validation')
  pyplot.legend()
  pyplot.show()

  # Make a prediction and plot results
  yhat_test = model.predict(test_X)

  pyplot.plot(test_y, label='test_y')
  pyplot.plot(yhat_test, label='yhat_test')
  for stgIndex in range(len(fld_stgs_scaled)):
    pyplot.axhline(y=fld_stgs_scaled[stgIndex], color=stg_colors[stgIndex], linestyle='-', label=stg_labels[stgIndex])
  pyplot.legend()
  pyplot.show()


  # Plot and evaluate prediction results
  axis_test = fig_test.add_subplot(1, 1, 1)
  axis_test.plot(test_y, label='Actual', linewidth=2)
  axis_test.plot(yhat_test, label='Predicted', linewidth=2.5, alpha=0.6, color='tab:pink')
  for stgIndex in range(len(fld_stgs_scaled)):
    axis_test.axhline(y=fld_stgs_scaled[stgIndex], color=stg_colors[stgIndex], linestyle='-', label=stg_labels[stgIndex])
  leg_test = axis_test.legend()
  

  yhat_valid = model.predict(X_validation)

  pyplot.plot(y_validation, label='y_validation')
  pyplot.plot(yhat_valid, label='yhat_valid')
  for stgIndex in range(len(fld_stgs_scaled)):
    pyplot.axhline(y=fld_stgs_scaled[stgIndex], color=stg_colors[stgIndex], linestyle='-', label=stg_labels[stgIndex])
  pyplot.legend()
  pyplot.show()

  axis_valid = fig_valid.add_subplot(1, 1, 1)
  axis_valid.plot(y_validation, label='Actual', linewidth=2)
  axis_valid.plot(yhat_valid, label='Predicted', linewidth=3, alpha=0.7, color='tab:pink')
  for stgIndex in range(len(fld_stgs_scaled)):
    axis_valid.axhline(y=fld_stgs_scaled[stgIndex], color=stg_colors[stgIndex], linestyle='-', label=stg_labels[stgIndex], linewidth=1)
  leg_valid = axis_valid.legend()


  df_validation = df_validation[~df_validation.isin([np.nan, np.inf, -np.inf]).any(1)] 

  global figs 
  global results
  global toReturn

  figs.append(fig_test)
  figs.append(fig_valid)


  try:
    i = 0
    dates = []
    resultsYhat = []
    resultsYvalid = []

    while (i < len(yhat_valid)-4):
      resultsRow = []
      max_yhat_valid = max( max(yhat_valid[i][0],yhat_valid[i+1][0]), max(yhat_valid[i+2][0],yhat_valid[i+3][0]) )
      max_y_validation = max( max(y_validation[i],y_validation[i+1]), max(y_validation[i+2],y_validation[i+3]) )
      
      dates.append(df_validation.iloc[i, df_validation.columns.get_loc('datetime')])
      resultsYhat.append(max_yhat_valid / fld_stgs_scaled[0])
      resultsYvalid.append(max_y_validation / fld_stgs_scaled[0])

      i += 4
    results.append(dates)
    results.append(resultsYhat)
    results.append(resultsYvalid)

    toReturn.append(figs)
    toReturn.append(results)

  except:
    print("ERROR OCCURRED IN PROCESSING OF VALIDATION RESULTS")
  

  X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[2]))
  
  # Invert scaling for forecast
  inv_yhat = concatenate((yhat_valid, X_validation[:, 1:]), axis=1)

  scaler = MinMaxScaler(feature_range=(0, 1)).fit(inv_yhat)

  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]

  # Invert scaling for actual
  y_validation = y_validation.reshape((len(y_validation), 1))
  inv_y = concatenate((y_validation, X_validation[:, 1:]), axis=1)

  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]

  # Calculate RMSE
  rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
  print('Test RMSE: %.3f' % rmse)

  




  # metrics = model.evaluate(test_X, test_y)
  # print(metrics)

  # print(yhat)
  # print(test_y)
  # print(inv_y)
  # print(inv_yhat)

# init("30day")

# if __name__ == '__main__':
#     main()