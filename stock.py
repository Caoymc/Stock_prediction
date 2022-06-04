import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# 0. Load data
df_data = pd.read_csv('AMD.csv')
#print(df_AAPLdata)
train_dates = pd.to_datetime(df_data['Date'])
#print(train_dates)

# 1. process the data
# variables for training
cols = list(df_data)[1:6]
#print(cols)
df_data = df_data[cols].astype(float)
print(len(df_data))

# (1) normalize the data
scaler = StandardScaler()
scaler = scaler.fit(df_data)
df_data_scaled = scaler.transform(df_data)

n_forcast = 90
df_train_scaled = df_data_scaled[:-n_forcast]
df_test_scaled = df_data_scaled[-n_forcast:]
print(len(df_train_scaled))
#print(df_AAPLdata_train_scaled)


#(3) reshape the data
train_X = []
train_Y = []
test_X = []
n_future = 1   #number of days to predict into the future
n_past = 14   #number of past days to use to predict the future (try different time range in the next steps, for now using 2 weeks of data)

for i in range(n_past, len(df_train_scaled) - n_future + 1):
    train_X.append(df_train_scaled[i - n_past:i, 0:df_train_scaled.shape[1]])
    train_Y.append(df_train_scaled[i + n_future - 1:i + n_future, 0])

for i in range(n_past, len(df_data_scaled) - n_future + 1):
    test_X.append(df_data_scaled[i - n_past:i, 0:df_data_scaled.shape[1]])

train_X, train_Y = np.array(train_X), np.array(train_Y)
test_X = np.array(test_X)
test_X = test_X[-n_forcast:]

print('train_X shape == {}.'.format(train_X.shape))
print('train_Y shape == {}.'.format(train_Y.shape))
print('test_X shape == {}.'.format(test_X.shape))


#Define the model
model = Sequential()
model.add(LSTM(64, activation = 'relu', input_shape = (train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(32, activation = 'relu', return_sequences=False))
model.add(Dropout(0.2))    # could make changes like how many to drop out
model.add(Dense(train_Y.shape[1]))

model.compile(optimizer = 'adam', loss = 'mse')
model.summary

# fit the model
# we could change some parameters here
history = model.fit(train_X, train_Y, epochs=3, batch_size=1, validation_split=0.1, verbose = 1)

# prediction on the train data
forecast = model.predict(test_X[-n_forcast:])

#rescale it back to original level
forecast_repeat = np.repeat(forecast, df_data.shape[1], axis = -1)
y_pred_future = scaler.inverse_transform(forecast_repeat)[:,0].T
print(y_pred_future.shape)

# graph to compare with the real data
y_valid = df_data['Open'].to_numpy()
y_valid = y_valid[-n_forcast*3:]
print(y_valid.shape)
y_pred_future = np.concatenate((y_valid[-n_forcast*3:-n_forcast], y_pred_future), axis=0)

daterange = list(np.arange(1,n_forcast*3+1))
plt.figure(figsize=(20,10))
plt.plot(daterange,y_pred_future, label = "prediction")
plt.plot(daterange,y_valid, label = "real data")
plt.legend()
plt.show()