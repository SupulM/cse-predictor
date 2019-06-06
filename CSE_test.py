import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import sys


def create_scaler(csv):
    dataset = pd.read_csv(csv)
    dataset = dataset.reindex(index=dataset.index[::-1])

    data_training = dataset.iloc[:, 1:2].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))

    sc.fit_transform(data_training)
    return sc


company_id = sys.argv[1]

train_path = 'datasets/' + company_id + '_train.csv'
test_path = 'datasets/' + company_id + '_test.csv'
model_path = 'saved/cse_' + company_id + '.h5'
no_steps = 60

dataset = pd.read_csv(train_path)
scaler = create_scaler(train_path)

from keras.models import load_model
regressor = load_model(model_path)

dataset_testing = pd.read_csv(test_path)
dataset_testing = dataset_testing.reindex(index=dataset_testing.index[::-1])

data_testing = dataset_testing.iloc[:, 1:2].values

dataset_total = pd.concat((dataset['Open (Rs.)'], dataset_testing['Open (Rs.)']), axis=0)
inputs = dataset_total[len(dataset_total) - len(data_testing) - no_steps:].values

inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []

for i in range(no_steps, no_steps + len(data_testing)):
    X_test.append(inputs[i - no_steps:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_prices = regressor.predict(X_test)
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

plt.plot(data_testing, color='red', label='Real ' + company_id + ' Stock Price')
plt.plot(predicted_stock_prices, color='green', label='Predicted ' + company_id + ' Stock Price')
plt.title('Stock price prediction for CSE (' + company_id + ')')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
