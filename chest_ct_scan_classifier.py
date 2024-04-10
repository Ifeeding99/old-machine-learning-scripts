import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, RandomFlip, RandomZoom, RandomRotation, BatchNormalization, Add, Activation, AveragePooling2D, Rescaling


train_path = 'C:/Users/User01/Desktop/chest_ct_scan_extracted/Data/train'
val_path = 'C:/Users/User01/Desktop/chest_ct_scan_extracted/Data/valid'
test_path = 'C:/Users/User01/Desktop/chest_ct_scan_extracted/Data/test'

img_size = 256
n_epochs = 20
batch_size = 32
activation = tf.nn.elu

train_dataset = image_dataset_from_directory(train_path,
                                             image_size = (img_size, img_size),
                                             label_mode = 'int',
                                             shuffle = True,
                                             batch_size = batch_size)

val_dataset = image_dataset_from_directory(val_path,
                                             image_size = (img_size, img_size),
                                             label_mode = 'int',
                                             batch_size = batch_size)

test_dataset = image_dataset_from_directory(test_path,
                                             image_size = (img_size, img_size),
                                             label_mode = 'int',
                                             batch_size = batch_size)

data_augmentation = tf.keras.Sequential([
    RandomFlip(input_shape = (img_size, img_size, 3)),
    RandomZoom(0.5),
    RandomRotation(0.5),
    Rescaling(1/255)
])

class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()
        self.conv1 = Conv2D(filters = self.n_filters, kernel_size = (3,3), strides = 1, padding = 'same')
        self.conv2 = Conv2D(filters = self.n_filters, kernel_size = (3,3), strides = 1, padding = 'same')


    def call(self, x):
        input_ = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = tf.keras.layers.Add()([x, input_])
        output = tf.keras.layers.Activation(activation)(x)
        return output


class Conv_Block_Size_Change(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()
        self.conv_to_add = Conv2D(filters = self.n_filters, kernel_size = (1,1), strides = 2, padding = 'same')
        self.conv1 = Conv2D(filters=self.n_filters, kernel_size=(3, 3), strides = 2, padding='same')
        self.conv2 = Conv2D(filters=self.n_filters, kernel_size=(3, 3), strides = 1, padding='same')

    def call(self, x):
        input_ = x
        residue_to_add = self.conv_to_add(input_)
        x = self.conv1(x)
        x = self.norm1(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = tf.keras.layers.Add()([x, residue_to_add])
        output = tf.keras.layers.Activation(activation)(x)
        return output



input_ = tf.keras.Input(shape = (img_size, img_size, 3))
input_1 = data_augmentation(input_)
x = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, activation = tf.nn.relu)(input_1)
x = MaxPool2D((2,2))(x)
x = Conv_Block(64)(x)
x = Conv_Block(64)(x)
x = Conv_Block(64)(x)
x = Conv_Block_Size_Change(128)(x)
x = Conv_Block(128)(x)
x = Conv_Block(128)(x)
x = Conv_Block_Size_Change(256)(x)
x = Conv_Block(256)(x)
x = Conv_Block(256)(x)
x = Conv_Block_Size_Change(512)(x)
x = Conv_Block(512)(x)
x = Conv_Block(512)(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
output = Dense(units = 4, activation = tf.nn.softmax)(x)



model = tf.keras.Model(inputs = input_, outputs = output)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])
history = model.fit(train_dataset,
          validation_data=test_dataset,
          epochs = n_epochs)

history = pd.DataFrame(history.history)
history.plot()
plt.grid()
plt.show()


