'''
With this script I try to classify movie reviews from the
tensorflow imbd dataset. The labels are 'positive' or 'negative'
'''

import tensorflow as tf
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN, Dense, Dropout, Embedding
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

vocabulary = 200000
(X_train, y_train), (X_test, y_test) = load_data()
# padding text
X_train = pad_sequences(X_train, maxlen = 200)
X_test = pad_sequences(X_test, maxlen = 200)

# building the model
input_ = tf.keras.Input(shape = (200,))
embedding_layer = Embedding(input_dim = vocabulary, output_dim = 300, mask_zero = True)(input_)
rnn_layer = GRU(units = 100, activation = 'tanh')(embedding_layer)
drop_1 = Dropout(0.3)(rnn_layer)
hidden_layer_1 = Dense(units = 64, activation = 'relu')(drop_1)
drop_2 = Dropout(0.5)(hidden_layer_1)
hidden_layer_2 = Dense(units = 32, activation = 'relu')(drop_2)
drop_3 = Dropout(0.5)(hidden_layer_2)
hidden_layer_3 = Dense(units = 16, activation = 'relu')(drop_3)
output = Dense(units = 1, activation = 'sigmoid')(hidden_layer_3)


model = tf.keras.Model(inputs = input_, outputs = output, name = 'MyModel')

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy'])

stop = EarlyStopping(min_delta = 0.001, patience = 3)

training = model.fit(X_train,
                     y_train,
                     validation_data = (X_test, y_test),
                     batch_size = 64,
                     epochs = 50,
                     callbacks = [stop])


pd.DataFrame(training.history).plot()
plt.grid()
plt.show()
model.save('imdb classifier')