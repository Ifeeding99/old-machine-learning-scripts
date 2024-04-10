import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomRotation, Flatten, Dropout, Dense



train_path = 'C:/Users/User01/Desktop/intel challenge/seg_train'
test_path = 'C:/Users/User01/Desktop/intel challenge/seg_test'

train_dataset = image_dataset_from_directory(train_path,
                                             image_size = (224,224),
                                             validation_split = 0.2,
                                             subset = 'training',
                                             label_mode = 'int',
                                             shuffle = True,
                                             batch_size = 32,
                                             seed = 42)

validation_dataset = image_dataset_from_directory(train_path,
                                                  image_size = (224,224),
                                                  validation_split = 0.2,
                                                  subset = 'validation',
                                                  batch_size = 42,
                                                  label_mode = 'int',
                                                  shuffle = True,
                                                  seed = 42)

test_dataset = image_dataset_from_directory(test_path,
                                            image_size = (224,224),
                                            label_mode = 'int',
                                            batch_size = 32)

stop = EarlyStopping(min_delta = 0.001, patience = 7)

pretrained_base = VGG19(include_top = False, weights = 'imagenet', input_shape = (224,224,3))
pretrained_base.trainable = False

# model
input_ = Input(shape = (224,224,3))
# preprocessing and data augmentation
r1 = RandomFlip()(input_)
r2 = RandomZoom(0.3,0.3)(r1)
r3 = RandomRotation(0.2)(r2)
p = preprocess_input(r3)
# body
vgg19_output = pretrained_base(p)
# head
flat = Flatten()(vgg19_output)
h1 = Dense(units = 512, activation = 'relu')(flat)
drop1 = Dropout(0.5)(h1)
h2 = Dense(units = 512, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(h2)
h3 = Dense(units = 64, activation = 'relu')(drop2)
drop3 = Dropout(0.5)(h3)
output_layer = Dense(units = 6, activation = 'softmax')(drop3)

model = Model(inputs = input_, outputs = output_layer)
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
training = model.fit(train_dataset,
                     validation_data = validation_dataset,
                     batch_size = 32,
                     epochs = 100,
                     callbacks = [stop])

history = pd.DataFrame(training.history)
history.plot()
plt.grid()
plt.show()
model.evaluate(test_dataset)
model.save('intel challenge with VGG19')