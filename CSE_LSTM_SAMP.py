import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/SAMP_train.csv')
dataset = dataset.reindex(index=dataset.index[::-1])

data_training = dataset.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(data_training)

X_train = []
y_train = []

for i in range(60, 1463):
    X_train.append(training_set_scaled[i-60:i, 0])
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

regressor.save('saved/cse_SAMP.h5')

from keras.models import load_model
regressor = load_model('saved/cse_SAMP.h5')

dataset_testing = pd.read_csv('datasets/SAMP_test.csv')
dataset_testing = dataset_testing.reindex(index=dataset_testing.index[::-1])

data_testing = dataset_testing.iloc[:, 1:2].values

dataset_total = pd.concat((dataset['Open (Rs.)'], dataset_testing['Open (Rs.)']), axis=0)
inputs = dataset_total[len(dataset_total) - len(data_testing) - 60:].values

inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 78):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_prices = regressor.predict(X_test)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices)

plt.plot(data_testing, color='red', label='Real SAMP Stock Price')
plt.plot(predicted_stock_prices, color='green', label='Predicted SAMP Stock Price')
plt.title('CSE Stock Price Trend Prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
