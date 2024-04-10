import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Rescaling, RandomFlip, RandomRotation, RandomZoom, Flatten

dir_path = 'C:/Users/User01/Desktop/colon cancer histology dataset'
log_path = 'logs'
if not os.path.exists(log_path):
    os.mkdir(log_path)

training_dataset = image_dataset_from_directory(dir_path,
                                                image_size = (150,150),
                                                validation_split = 0.2,
                                                subset = 'training',
                                                shuffle = True,
                                                label_mode = 'int',
                                                seed = 7)

validation_dataset = image_dataset_from_directory(dir_path,
                                                image_size = (150,150),
                                                validation_split = 0.2,
                                                subset = 'validation',
                                                shuffle = True,
                                                label_mode = 'int',
                                                seed = 7)

early_stop = EarlyStopping(min_delta = 0.02, patience = 3)
tensorboard = TensorBoard(log_dir = log_path)

# creating model
input_ = Input(shape = (150,150,3))
resc = Rescaling(1/255)(input_)
ran_flip = RandomFlip()(resc)
ran_rot = RandomRotation(0.5)(ran_flip)
ran_zoom = RandomZoom(0.3)(ran_rot)
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')(ran_zoom)
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = MaxPool2D(pool_size = (2,2))(x)
x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = MaxPool2D(pool_size = (2,2))(x)
x = Conv2D(filters = 256, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = Flatten()(x)
x = Dense(units = 64, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(units = 64, activation = 'relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(units = 8, activation = 'softmax')(x)

model = Model(inputs = input_, outputs = output_layer)
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

training = model.fit(training_dataset,
                     validation_data = validation_dataset,
                     batch_size = 32,
                     epochs = 100,
                     callbacks = [tensorboard, early_stop])
