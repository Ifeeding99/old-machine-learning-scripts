import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5'
train_path = 'C:/Users/User01/Desktop/dogs-vs-cats/train'
pretrained_base = hub.KerasLayer(url, trainable = False)
#pretrained_base = keras.applications.InceptionV3(include_top=False, input_shape=(299,299,3))
pretrained_base.trainable = False

train = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory(train_path,
                                          class_mode = 'binary',
                                          batch_size=32,
                                          target_size=(299,299))

model = tf.keras.Sequential([
    pretrained_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
)

t = model.fit(train_dataset,
              batch_size = 32,
              epochs = 60)
h = pd.DataFrame(t.history)
h.plot()
plt.grid()
plt.show()
