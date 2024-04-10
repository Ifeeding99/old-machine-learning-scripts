
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import RandomZoom, RandomFlip, RandomRotation,Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras import Input, Model

file_path = 'C:/Users/User01/Desktop/foto f e v'
train_dataset = image_dataset_from_directory(file_path,
                                             image_size = (64,64),
                                             shuffle = True,
                                             label_mode = 'binary',
                                             )
# model
input_ = Input(shape=(64,64,3))
x = RandomZoom(0.2)(input_)
x = RandomFlip()(x)
x = RandomRotation(0.5)(x)
x = Conv2D(filters = 64, activation = 'relu', kernel_size = (3,3))(x)
x = MaxPool2D()(x)
x = Conv2D(filters = 64, activation = 'relu', kernel_size = (3,3))(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(units = 32, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(units = 32, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(units = 16, activation = 'relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(units = 1, activation = 'sigmoid')(x)

model = Model(inputs = input_, outputs = output_layer)


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy'])

training = model.fit(train_dataset,
                     batch_size = 3,
                     epochs = 1000)
history = pd.DataFrame(training.history)
history.plot()
plt.grid()
plt.show()
model.save('f_v_predictor')
