import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomRotation, Rescaling, Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras import Input,Model
from tensorflow.keras.callbacks import EarlyStopping

train_path = 'C:/Users/User01/Desktop/intel challenge/seg_train'
test_path = 'C:/Users/User01/Desktop/intel challenge/seg_test'

train_dataset = image_dataset_from_directory(train_path,
                                             validation_split = 0.2,
                                             subset = 'training',
                                             image_size = (75,75), # InceptionV3 wants images with size 75x75 at minimum
                                             shuffle = True,
                                             batch_size = 32,
                                             label_mode = 'int',
                                             seed = 42)

validation_dataset = image_dataset_from_directory(train_path,
                                                  validation_split = 0.2,
                                                  subset = 'validation',
                                                  image_size = (75,75), # InceptionV3 wants images with size 75x75 at minimum
                                                  shuffle = True,
                                                  batch_size = 32,
                                                  label_mode = 'int',
                                                  seed = 42)

test_dataset = image_dataset_from_directory(test_path,
                                            image_size = (75,75), # InceptionV3 wants images with size 75x75 at minimum
                                            batch_size = 32,
                                            shuffle = True,
                                            label_mode = 'int',
                                            seed = 42)

pretrained_base = InceptionV3(include_top = False,
                              weights = 'imagenet',
                              input_shape = (75,75,3))
pretrained_base.trainable = False

# my model
input_ = Input(shape = (75,75,3))
# body
rescaled = Rescaling(1/255)(input_)
rotation = RandomRotation(0.2)(rescaled)
zoom = RandomZoom(0.3,0.3)(rotation)
flip = RandomFlip()(zoom)
base = pretrained_base(flip)
# head
flat_layer = Flatten()(base)
h1_layer = Dense(units = 128, activation = 'relu')(flat_layer)
drop1 = Dropout(0.5)(h1_layer)
h2_layer = Dense(units = 128, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(h2_layer)
h3_layer = Dense(units = 128, activation = 'relu')(drop2)
drop3 = Dropout(0.5)(h3_layer)
output = Dense(units = 6, activation = 'softmax')(drop3)

model = Model(inputs = input_, outputs = output)
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

stop = EarlyStopping(min_delta = 0.001, patience = 5)

training = model.fit(train_dataset,
                     validation_data = validation_dataset,
                     batch_size = 32,
                     epochs = 100,
                     callbacks = [stop])

history = pd.DataFrame(training.history).plot()
plt.grid()
plt.show()

model.evaluate(test_dataset) # it had around 0.72 accuracy
model.save('model for intel challenge with InceptionV3')