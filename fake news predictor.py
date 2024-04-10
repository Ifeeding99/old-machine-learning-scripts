import pandas as pd
import matplotlib.pyplot as plt
import re
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TextVectorization, Embedding, GRU, LSTM, SimpleRNN, Dropout, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

train_df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/fake-news/train.csv')
train_df = train_df.dropna()
train_df.pop('id')
y_train = train_df.pop('label')

test_df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/fake-news/test.csv')
test_df = test_df.dropna()
test_df.pop('id')
y_test = test_df.pop('label')

# removing special characters and stuff like that
# training data
new_text = []
for el in train_df.title:
    n_t = re.sub('[^A-Za-z-\s]','',el)
    new_text.append(n_t)
train_df.title = new_text

new_text = []
for el in train_df.author:
    n_t = re.sub('[^A-Za-z-\s]','',el)
    new_text.append(n_t)
train_df.author = new_text

new_text = []
for el in train_df.text:
    n_t = re.sub('[^A-Za-z-\s]','',el)
    new_text.append(n_t)
train_df.text = new_text

# test data
new_text = []
for el in test_df.title:
    n_t = re.sub('[^A-Za-z-\s]','',el)
    new_text.append(n_t)
test_df.title = new_text

new_text = []
for el in test_df.author:
    n_t = re.sub('[^A-Za-z-\s]','',el)
    new_text.append(n_t)
test_df.author = new_text

new_text = []
for el in test_df.text:
    n_t = re.sub('[^A-Za-z-\s]','',el)
    new_text.append(n_t)
test_df.text = new_text

vocab_size = 20000
encoder = TextVectorization(max_tokens = vocab_size, output_mode = 'int', output_sequence_length = 300)
encoder.adapt(train_df.title)


train_df.author = encoder(train_df.author)
train_df.title = encoder(train_df.title)


test_df.author = encoder(test_df.author)
test_df.title = encoder(test_df.title)


train_df.pop('text')
test_df.pop('text') # I dropped the text section because:
# a) the articles are too long
# b) I cannot increase too much the vocabulary size
# c) (and most important) the model literally breaks because it hasn't got sufficient resources
# model
input1 = Input(shape=(300))
input2 = Input(shape=(300))
embed_layer_1 = Embedding(input_dim = vocab_size, output_dim = 256, mask_zero = True)(input1)
embed_layer_2 = Embedding(input_dim = vocab_size, output_dim = 256, mask_zero = True)(input2)
rnn_layer_1 = GRU(units = 128, activation = 'tanh')(embed_layer_1)
rnn_layer_2 = GRU(units = 128, activation = 'tanh')(embed_layer_2)
concat = Concatenate()([rnn_layer_1, rnn_layer_2])
h1 = Dense(units = 64, activation = 'relu')(concat)
drop1 = Dropout(0.6)(h1)
h2 = Dense(units = 64, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(h2)
output_layer = Dense(units = 1, activation = 'sigmoid')(h2)
model = Model(inputs = [input1, input2], outputs = output_layer)

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy'])

stop = EarlyStopping(min_delta = 0.001, patience = 3)

model.fit([train_df.author, train_df.title], y_train,
          epochs = 7,
          batch_size = 32,
          validation_split = 0.2,
          callbacks = [stop])

#model.evaluate(test_df, y_test)



