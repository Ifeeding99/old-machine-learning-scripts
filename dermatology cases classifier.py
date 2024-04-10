'''
This code trains a model that recognizes different pathologies of the skin.
The dataset is taken from kaggle
https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

About Dataset
This set consists of 2357 images of malignant and benign oncological diseases,
which were formed from The International Skin Imaging Collaboration (ISIC).
All images were sorted according to the classification taken with ISIC,
and all subsets were divided into the same number of images,
with the exception of melanomas and moles, whose images are slightly dominant.

The data set contains the following diseases:

- actinic keratosis
- basal cell carcinoma
- dermatofibroma
- melanoma
- nevus
- pigmented benign keratosis
- seborrheic keratosis
- squamous cell carcinoma
- vascular lesion
'''

import time
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, RandomFlip, RandomRotation, Rescaling
from tensorflow.keras.applications.inception_v3 import preprocess_input

train_path = 'C:/Users/User01/Desktop/dermatology images classification/Train'
test_path = 'C:/Users/User01/Desktop/dermatology images classification/Test'
img_height = 224
img_width = 224

'''
# Uncomment to view the images
img = cv2.imread('C:/Users/User01/Desktop/dermatology images classification/Train/basal cell carcinoma/ISIC_0024550.jpg')
img = cv2.resize(img,(img_width, img_height))
img = cv2.cvtColor(img,code=cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
'''

'''
pretrained_model = tf.keras.applications.inception_v3.InceptionV3(include_top = True,
                                                                  classifier_activation = 'softmax',
                                                                  input_shape = (224,224,3),
                                                                  classes = 9,
                                                                  weights = None)
'''

pretrained_base = tf.keras.applications.inception_v3.InceptionV3(include_top = False, input_shape =(224,224,3))
pretrained_base.trainable = False


train_dataset = image_dataset_from_directory(train_path,
                                             label_mode = 'categorical', # used for sparse categorical data
                                             validation_split = 0.2,
                                             image_size = (img_width, img_height),
                                             shuffle = True,
                                             seed = 0,
                                             batch_size = 32,
                                             subset = 'training')

validation_dataset = image_dataset_from_directory(train_path,
                                                  label_mode = 'categorical',
                                                  image_size = (img_width, img_height),
                                                  validation_split = 0.2,
                                                  seed = 0,
                                                  shuffle = True,
                                                  batch_size = 32,
                                                  subset = 'validation')

test_dataset = image_dataset_from_directory(test_path,
                                           image_size = (img_width, img_height),
                                            label_mode = 'categorical',
                                            seed = 0,
                                            batch_size = 32)

def preprocess_dataset(im,labels):
    return preprocess_input(im),labels

model1 = Sequential([
    Rescaling(1/255),
    # Data augmentation
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(factor = (-2, 2)),
    # Convolutional layers (base)
    Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    MaxPool2D(),
    Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'),
    MaxPool2D(),
    Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'),
    MaxPool2D(),
    # Head
    Flatten(),
    Dense(units = 128, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 128, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 9, activation = 'softmax')

])

model2 = Sequential([
    # base
    Rescaling(1/255),
    pretrained_base,
    MaxPool2D(),
    # head
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(9, activation = 'softmax')
])

cnnmodel = Sequential([
    Rescaling(1/255),
    Conv2D(filters= 32, kernel_size = (3,3), activation = 'relu'),
    Flatten(),
    Dense(9,activation = 'softmax')
])

cnnmodel.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

train_dataset = train_dataset.map(preprocess_dataset)
validation_dataset = validation_dataset.map(preprocess_dataset)
test_dataset = test_dataset.map(preprocess_dataset)
start_training = time.time()

training = cnnmodel.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = 20
)

end_training = time.time()
print(f'training took {round(end_training - start_training, 3)} s\n')
history = pd.DataFrame(training.history)
history.plot()
plt.grid()
plt.show()

print('TEST')
cnnmodel.evaluate(test_dataset)
