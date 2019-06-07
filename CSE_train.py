import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys


def create_scaler(csv):
    dataset = pd.read_csv(csv)
    dataset = dataset.reindex(index=dataset.index[::-1])

    data_training = dataset.iloc[:, 1:2].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))

    training_set_scaled = sc.fit_transform(data_training)
    return sc


def train_regressor(csv, steps):
    dataset = pd.read_csv(csv)
    dataset = dataset.reindex(index=dataset.index[::-1])

    data_training = dataset.iloc[:, 1:2].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))

    training_set_scaled = sc.fit_transform(data_training)

    X_train = []
    y_train = []

    for i in range(steps, len(dataset)):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    regressor = Sequential()

    # First LSTM Layer
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(rate=0.2))

    # Second LSTM Layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Third LSTM Layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Fourth LSTM Layer
    regressor.add(LSTM(units=50, return_sequences=False))
    regressor.add(Dropout(rate=0.2))

    # Output layer
    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=100, batch_size=32)

    return regressor


company_id = sys.argv[1]

train_path = 'datasets/' + company_id + '_train.csv'
test_path = 'datasets/' + company_id + '_test.csv'
no_steps = 60

dataset = pd.read_csv(train_path)
scaler = create_scaler(train_path)

regressor = train_regressor(train_path, no_steps)

model_name = 'saved/cse_' + company_id + '.h5'
regressor.save(model_name)
print('Model trained and saved at : ' + model_name)