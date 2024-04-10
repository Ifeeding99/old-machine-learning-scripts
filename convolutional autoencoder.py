import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose

(X_train, y_train),(X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential([
            Conv2D(input_shape = [28,28,1], filters = 32, kernel_size = (3,3), padding = 'same', strides = 2, activation = 'relu'),
            Conv2D(16, padding = 'same', kernel_size = (3,3), strides = 2, activation = 'relu'),
            Conv2D(8, padding='same', kernel_size = (3,3), strides=1, activation='relu')
        ])
        self.decoder = Sequential([
            Conv2DTranspose(8, padding = 'same', strides = 2, activation = 'relu',kernel_size = (3,3)),
            Conv2DTranspose(16, padding='same', strides=2, activation='relu',kernel_size = (3,3)),
            Conv2DTranspose(filters = 32, padding='same', strides=1, activation='relu',kernel_size = (3,3)),
            Conv2D(1, activation = 'sigmoid', padding = 'same',kernel_size = (3,3))

        ])

    def call(self,x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output

model = AutoEncoder()
model.compile(optimizer = 'adam',
              loss = 'mean_squared_error')


model.fit(X_train, X_train,
          validation_data = (X_test, X_test),
          epochs = 10,
          batch_size=32)

to_generate = np.expand_dims(X_test[14], axis = 0)
generated = model(to_generate).numpy()
generated = np.reshape(generated, (28,28,1)) # output dim is (1,28,28) 1 is the batch size
fig, ax = plt.subplots(1,2)
ax[0].imshow(X_test[14], cmap = 'Greys')
ax[0].set_title('Original')
ax[1].imshow(generated, cmap = 'Greys')
ax[1].set_title('Generated')
plt.show()