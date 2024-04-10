import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets.cifar10 import load_data

(X_train, y_train), (X_test, y_test) = load_data()

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)
y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)

from tensorflow.keras.callbacks import EarlyStopping

stop = EarlyStopping(min_delta = 0.001,
                     patience = 5,
                     restore_best_weights = True)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization,RandomFlip, Rescaling, RandomRotation, GlobalAveragePooling2D,Flatten, Dropout, Dense

model = Sequential([
                    Rescaling(1/255),
                    RandomFlip(),
                    RandomRotation(0.2),

                    Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
                    MaxPool2D(),
                    BatchNormalization(),

                    Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
                    MaxPool2D(),
                    BatchNormalization(),

                    Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'),
                    MaxPool2D(),
                    BatchNormalization(),

                    Flatten(),
                    Dense(units = 128, activation = 'relu'),
                    Dropout(0.5),
                    BatchNormalization(),

                    Dense(units = 128, activation = 'relu'),
                    Dropout(0.5),
                    BatchNormalization(),

                    Dense(units = 32, activation = 'relu'),
                    Dropout(0.5),
                    BatchNormalization(),

                    Dense(units = 10, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

training = model.fit(
    X_train,
    y_train,
    validation_split = 0.2,
    epochs = 500,
    batch_size = 64,
    callbacks = [stop]
)

history = pd.DataFrame(training.history)
history.plot()
plt.grid()
plt.show()

model.evaluate(X_test, y_test)