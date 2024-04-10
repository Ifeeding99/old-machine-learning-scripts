import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Rescaling, RandomRotation, RandomFlip, RandomZoom
from tensorflow.keras import Input, Model

dir_path = 'C:/Users/User01/Desktop/brain tumors/Training'

img_size = 150

train_dataset = image_dataset_from_directory(dir_path,
                                             image_size = (img_size,img_size),
                                             seed = 7,
                                             shuffle = True,
                                             batch_size = 32,
                                             label_mode = 'int',
                                             validation_split = 0.2,
                                             subset = 'training')

val_dataset = image_dataset_from_directory(dir_path,
                                           image_size = (150,150),
                                             seed = 7,
                                             shuffle = True,
                                             batch_size = 32,
                                             label_mode = 'int',
                                             validation_split = 0.2,
                                             subset = 'validation')

# model
input_ = Input(shape = (img_size,img_size,3))
x = Rescaling(1/255)(input_)
x = Conv2D(32,(3,3),activation = tf.nn.relu)(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(64,(3,3),activation = tf.nn.relu)(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(128,(3,3),activation = tf.nn.relu)(x)
x = MaxPool2D((2,2))(x)
x = Flatten()(x)
x = Dense(256, activation = tf.nn.relu)(x)
x = Dense(128, activation = tf.nn.relu)(x)
output = Dense(4, activation = tf.nn.softmax)(x)

model = Model(inputs = input_, outputs = output)
model.summary()
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])
model.fit(train_dataset,
          validation_data = val_dataset,
          epochs = 15)