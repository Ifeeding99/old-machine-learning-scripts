import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.keras.layers import LSTM, Conv1D, MaxPool1D, GlobalMaxPool1D, Flatten, Dropout, Dense, Normalization

train_size = 4561
path = 'C:/Users/User01/Downloads/dati prezzi 5000.xlsx'
df = pd.read_excel(path)
df = df.dropna()

df_2 = pd.read_excel(path, parse_dates = True, index_col = 'Exchange Date')
df_2 = df.dropna()
df_2.Net.plot(color = 'blue', marker = '.')
plt.grid()
plt.show()

train_size = round(len(df.Open)*0.8)
val_size = round(train_size+(len(df.Open)-train_size)/2)
seq_len = 30
X = np.array(df.Open)
X = np.expand_dims(X, axis = 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = np.array(df.Label)

for i,el in enumerate(y):
    if el > 0:
        y[i] = 1
    else:
        y[i] = 0

y = np.expand_dims(y,axis = 1)
X_train = X[:train_size]
y_train = y[seq_len:train_size]
X_val = X[train_size:val_size]
y_val = y[train_size + seq_len:val_size]
X_test = X[val_size:]
y_test = y[val_size + seq_len:]

dataset_train = timeseries_dataset_from_array(X_train, targets = y_train, sequence_length = seq_len, sequence_stride = 1, batch_size = 64, shuffle = False)
dataset_val= timeseries_dataset_from_array(X_val, targets = y_val, sequence_length = 30, sequence_stride = 1, batch_size = 64, shuffle = False)
dataset_test = timeseries_dataset_from_array(X_test, targets = y_test, sequence_length = 30, sequence_stride = 1, batch_size = 64, shuffle = False)

# model
input_ = Input(shape = (30,1))
x = LSTM(64, return_sequences = True)([input_])
x = LSTM(512, return_sequences = True)(x)
x = LSTM(512, return_sequences = True)(x)
x = LSTM(256, return_sequences = False)(x)
x = Dense(128, activation = tf.nn.relu)(x)
x = Dense(256, activation = tf.nn.relu)(x)
x = Dense(1, activation = tf.nn.sigmoid)(x)
model = Model(inputs = input_, outputs = x)

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])

model.fit(dataset_train,
          validation_data = dataset_val,
          epochs = 1000)

print('EVALUATION')
model.evaluate(dataset_test)