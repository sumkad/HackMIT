# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import numpy as np
import pandas as pd
# Requires all tensorflow dependencies
try:
    from tensorflow import keras
    # import tensorflow.keras as keras
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

from getJson import getData

userLatitude = 0.0
userLongitude = 0.0
requestYear = 2022
requestMonth = 5
requestDay = 1
predIntervalLength = 30

fig_test = Figure()
fig_valid = Figure()

figs = []  # fig_test, fig_valid
results = []  # datetime, yhat_valid, y_validation
toReturn = []


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
    print("Cols")
    print(cols)
    print("Names")
    print(names)
    agg = concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg
    return agg

def init(latitude, longitude, year, month, day):
    global userLatitude
    global userLongitude
    global requestYear
    global requestMonth
    global requestDay

    userLatitude = latitude
    userLongitude = longitude
    requestYear = year
    requestMonth = month
    requestDay = day
    loadDataFrame()

def loadDataFrame():
    global userLatitude
    global userLongitude
    global requestYear
    global requestMonth
    global requestDay

    # Load dataset
    dataset = pd.read_csv("trainingData.json")
    # datetime
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'], errors='coerce')

    # numeric
    dataset['Site'] = pd.to_numeric(dataset['Site'])
    dataset['Latitude'] = pd.to_numeric(dataset['Latitude'])
    dataset['Longitude'] = pd.to_numeric(dataset['Longitude'])
    dataset['Temperature'] = pd.to_numeric(dataset['Temperature'])
    dataset['Conductance'] = pd.to_numeric(dataset['Conductance'])
    dataset['Dissolved_oxygen'] = pd.to_numeric(dataset['Dissolved_oxygen'])
    dataset['PH'] = pd.to_numeric(dataset['PH'])
    dataset['Turbidity'] = pd.to_numeric(dataset['Turbidity'])
    
    dataset = dataset.drop("Site", axis=1)
    dataset = dataset.drop("Latitude", axis=1)
    dataset = dataset.drop("Longitude", axis=1)

    # Replace all NaNs with value from previous row, the exception being Gage_height;
    # Only consider rows with valid Gage_height values
    dataset = dataset[dataset['Conductance'].notna()]

    for col in dataset:
        dataset[col].fillna(method='pad', inplace=True)

    # # Move Conductance to last column, as the value we are predicting
    # dataset = dataset[['DateTime'] + [c for c in dataset if c not in ['Conductance']] + ['Conductance']]

    
    # Validation data
    last_date = date(requestYear, requestMonth, requestDay)
    df_validation = dataset.copy()
    d1 = last_date
    d2 = last_date + timedelta(days=predIntervalLength)
    df_validation = df_validation.drop(
        df_validation[df_validation['DateTime'].dt.date < d1].index)

    df_validation = df_validation.drop(
        df_validation[df_validation['DateTime'].dt.date > d2].index)


    values = dataset.copy().drop('DateTime', axis=1).values
    print("Relevant Columns:")
    print(dataset.columns)
    
    print("Values:")
    print(values)
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    print(reframed.head())

    # Repeat for validation data
    df_validation_relevant = df_validation.copy()
    df_validation_relevant = df_validation_relevant.drop('DateTime', 1)
    validation_vals = df_validation_relevant.values
    validation_vals = validation_vals.astype('float32')
    validation_scaled = scaler.fit_transform(validation_vals)
    validation_reframed = series_to_supervised(validation_scaled, 1, 1)

    makePredictions(dataset, reframed, validation_reframed, df_validation)



def makePredictions(dataset, reframed, validation_reframed, df_validation):
    scaler = MinMaxScaler(feature_range=(0, 1))

    min_conduct = dataset['Conductance'].min()
    mean_conduct = dataset['Conductance'].mean()
    max_conduct = dataset['Conductance'].max()

    df_conductance_levels = pd.DataFrame(
        {'Conductance': [min_conduct, mean_conduct, max_conduct]})

    conductance_scaled = scaler.fit_transform(df_conductance_levels.values)
    conductance_scaled = conductance_scaled[1][0]

    conductance_levels_scaled = [(1.5 * conductance_scaled), (conductance_scaled), (
        0.5 * conductance_scaled), (0.25 * conductance_scaled)]
    conductance_colors = ['r', 'tab:orange', 'y', 'g']
    conductance_labels = ['150% Mean', 'Mean Conductance', '50% Mean', '25% Mean']

    df_validation.append(pd.Series(), ignore_index=True)
    # Set last row to mean
    # df_validation.iloc[-1, df_validation.columns.get_loc('Conductance')] = mean_conduct_valid

    print("VALIDATION MIN AND MAX")
    min_conduct_valid = df_validation['Conductance'].min()
    print(min_conduct_valid)
    mean_conduct_valid = df_validation['Conductance'].mean()
    print(mean_conduct_valid)
    max_conduct_valid = df_validation['Conductance'].max()
    print(max_conduct_valid)

    # Split into train and test sets
    print("REFRAMED COLUMNS:")
    print(reframed.columns)
    values = reframed.values
    n_train_hours = math.floor(len(dataset.index) * 0.7)
    train = values[n_train_hours:, :]
    test = values[:n_train_hours, :]

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
    model.add(Dense(1, activation=keras.activations.sigmoid))
    model.compile(loss='mae', optimizer='rmsprop', metrics=['mse', 'mae'])
    # Fit network
    history = model.fit(train_X, train_y, epochs=55, batch_size=100, validation_data=(test_X, test_y), verbose=2,  shuffle=False)  # validation_split= 0.2)

    # Plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    # Make a prediction and plot results
    print("TEST_X")
    print(test_X.shape)
    yhat_test = model.predict(test_X)
    print("YHAT_TEST")
    print(yhat_test)

    pyplot.plot(test_y, label='test_y')
    pyplot.plot(yhat_test, label='yhat_test')
    for condLevelIndex in range(len(conductance_levels_scaled)):
        pyplot.axhline(y=conductance_levels_scaled[condLevelIndex], color=conductance_colors[condLevelIndex],
                       linestyle='-', label=conductance_labels[condLevelIndex])
    pyplot.legend()
    pyplot.savefig("test.png")
    pyplot.show()

    # Plot and evaluate prediction results
    axis_test = fig_test.add_subplot(1, 1, 1)
    axis_test.plot(test_y, label='Actual', linewidth=2)
    axis_test.plot(yhat_test, label='Predicted', linewidth=2.5, alpha=0.6, color='tab:pink')
    for condLevelIndex in range(len(conductance_levels_scaled)):
        axis_test.axhline(y=conductance_levels_scaled[condLevelIndex], color=conductance_colors[condLevelIndex], linestyle='-', label=conductance_labels[condLevelIndex])
    leg_test = axis_test.legend()

    yhat_valid = model.predict(X_validation)

    pyplot.plot(y_validation, label='y_validation')
    pyplot.plot(yhat_valid, label='yhat_valid')
    for condLevelIndex in range(len(conductance_levels_scaled)):
        pyplot.axhline(y=conductance_levels_scaled[condLevelIndex], color=conductance_colors[condLevelIndex], linestyle='-', label=conductance_labels[condLevelIndex])
    pyplot.legend()
    pyplot.savefig("validation.png")
    pyplot.show()

    axis_valid = fig_valid.add_subplot(1, 1, 1)
    axis_valid.plot(y_validation, label='Actual', linewidth=2)
    axis_valid.plot(yhat_valid, label='Predicted',
                    linewidth=3, alpha=0.7, color='tab:pink')
    for condLevelIndex in range(len(conductance_levels_scaled)):
        axis_valid.axhline(y=conductance_levels_scaled[condLevelIndex], color=conductance_colors[condLevelIndex], linestyle='-', label=conductance_labels[condLevelIndex], linewidth=1)
    leg_valid = axis_valid.legend()

    df_validation = df_validation[~df_validation.isin(
        [np.nan, np.inf, -np.inf]).any(1)]

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
            max_yhat_valid = max(max(yhat_valid[i][0], yhat_valid[i+1][0]), max(yhat_valid[i+2][0], yhat_valid[i+3][0]))
            max_y_validation = max(max(y_validation[i], y_validation[i+1]), max(y_validation[i+2], y_validation[i+3]))

            dates.append(
                df_validation.iloc[i, df_validation.columns.get_loc('DateTime')])
            resultsYhat.append(max_yhat_valid / conductance_levels_scaled[0])
            resultsYvalid.append(max_y_validation / conductance_levels_scaled[0])

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
    inv_yhat = inv_yhat[:, 0]

    # Invert scaling for actual
    y_validation = y_validation.reshape((len(y_validation), 1))
    inv_y = concatenate((y_validation, X_validation[:, 1:]), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


init(0.0, 0.0, 2022, 5, 1)
