import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import time
import numpy as np

path = 'cartella_test_optimizers'
NAME = 'my_model_selu_improved'
if not(os.path.exists(path)):
    os.makedirs(path)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # to handle problems with downloading cifar10 dataset

(X_train, y_train), (X_val, y_val) = cifar10.load_data()

norm = tf.keras.layers.Normalization()
norm.adapt(X_train)
X_train = norm(X_train)
X_val = norm(X_val)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
for _ in range (20):
    model.add(tf.keras.layers.AlphaDropout(0.4))
    model.add(tf.keras.layers.Dense(units = 100, activation = 'selu',
                                kernel_initializer='lecun_normal'))

model.add(tf.keras.layers.AlphaDropout(0.4))
model.add(tf.keras.layers.Dense(units = 10, activation='softmax'))

tensorboard = TensorBoard(log_dir = path+'/'+NAME)
early_stop = EarlyStopping(min_delta = 0.01, patience = 5, restore_best_weights = True)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.9, nesterov=True)
model.compile(optimizer=optimizer,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              metrics = ['accuracy'])
model.fit(X_train, y_train,
          validation_data = [X_val, y_val],
          batch_size = 512,
          epochs = 500,
          callbacks = [tensorboard, early_stop])
