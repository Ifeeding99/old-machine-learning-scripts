import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, Rescaling, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import time

training_path = 'C:/Users/User01/Desktop/chihuahua-muffin/train'
val_path = 'C:/Users/User01/Desktop/chihuahua-muffin/test'
img_size = 224

training_dataset = image_dataset_from_directory(training_path,
                                                image_size = (img_size,img_size),
                                                label_mode = 'int',
                                                shuffle = True,
                                                batch_size = 64)

validation_dataset = image_dataset_from_directory(training_path,
                                                image_size = (img_size,img_size),
                                                label_mode = 'int',
                                                batch_size = 64)

class myModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rescale = Rescaling(1/255)
        self.conv1 = Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', strides = 1, activation = 'relu')
        self.pool = MaxPool2D((2,2))
        self.conv2 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same')
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')
        self.conv4 = Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', strides = 1, padding = 'same')
        self.flatten = Flatten()
        self.d1 = Dense(256, activation = 'relu')
        self.d2 = Dense(256, activation = 'relu')
        self.d3 = Dense(1, activation = 'sigmoid')

    def call(self, x):
        x = self.rescale(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

model = myModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
criterion = tf.keras.losses.BinaryCrossentropy()
acc_func_training = tf.keras.metrics.BinaryAccuracy()
acc_func_val = tf.keras.metrics.BinaryAccuracy()
num_epochs = 30

@tf.function
def training_step(x,y):
    with tf.GradientTape() as tape:
        pred = model(x, training = True)
        loss = criterion(y, pred)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_func_training.update_state(y, pred)
    return loss

start = time.time()

for epoch in range(num_epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(training_dataset):
        loss = training_step(X_batch, y_batch)

    accuracy_training = acc_func_training.result()

    for batch_val_idx, (X_batch_val, y_batch_val) in enumerate(validation_dataset):
        pred_val = model(X_batch_val, training = False)
        val_loss = criterion(y_batch_val, pred_val)
        acc_func_val.update_state(y_batch_val, pred_val)

    accuracy_val = acc_func_val.result()
    print(f'Epoch {epoch+1}/{num_epochs} ')
    print( f'training_loss: {round(loss.numpy(),3)}  ')
    print(f'training_accuracy: {accuracy_training}  ')
    print(f'val_loss: {val_loss} ')
    print(f'val_accuracy: {accuracy_val}')
    print('\n \n \n')
    acc_func_training.reset_state()
    acc_func_val.reset_state()

end = time.time()

print(f'It has taken { end - start}   s')