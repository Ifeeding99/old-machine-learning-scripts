import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Rescaling, Dropout, Flatten, Dense
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping

training_path = 'C:/Users/User01/Desktop/breast histopathology dataset'

train_dataset = image_dataset_from_directory(training_path,
                                             validation_split = 0.2,
                                             subset = 'training',
                                             color_mode = 'rgb',
                                             image_size = (50,50),
                                             label_mode = 'binary',
                                             shuffle = True,
                                             seed = 42)

validation_dataset = image_dataset_from_directory(training_path,
                                                  subset = 'validation',
                                                  validation_split = 0.2,
                                                  color_mode = 'rgb',
                                                  shuffle = True,
                                                  image_size = (50,50),
                                                  label_mode = 'binary',
                                                  seed = 42)

print(train_dataset.class_names) # the first is encoded as 0, the second is encoded as 1
# model
input_ = Input(shape = (50,50,3))
x = Rescaling(1/255)(input_)
x = RandomFlip()(x)
x = RandomRotation(0.5)(x)
x = RandomZoom(0.3)(x)
x = Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = MaxPool2D()(x)
x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(units = 64, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(units = 32, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(units = 16, activation = 'relu')(x)
x = Dropout(0.5)(x)
output = Dense(units = 1, activation = 'sigmoid')(x)
model = Model(inputs = input_, outputs = output)

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy'])

stop = EarlyStopping(min_delta = 0.001, patience = 3)

training = model.fit(train_dataset,
                     validation_data = validation_dataset,
                     batch_size = 64,
                     epochs = 200,
                     callbacks = [stop])

history = pd.DataFrame(training.history)
history.plot()
plt.grid()
plt.show()




