import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Dense, GlobalAvgPool2D, Rescaling, RandomFlip, RandomZoom, RandomRotation
from tensorflow.keras.applications.xception import Xception, preprocess_input

path = 'C:/Users/User01/Desktop/skin_cancers/training'

train_dataset = image_dataset_from_directory(path,
                                             image_size = (300,300),
                                             label_mode = 'int',
                                             subset = 'training',
                                             validation_split = 0.2,
                                             batch_size = 32,
                                             seed = 7)

val_dataset = image_dataset_from_directory(path,
                                           image_size = (300,300),
                                           label_mode = 'int',
                                           subset = 'validation',
                                           validation_split = 0.2,
                                           batch_size = 32,
                                           seed = 7)

pretrained_base = Xception(include_top = False, input_shape = (300,300,3))
pretrained_base.trainable = False


# model
class myModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rescale = Rescaling(1/255)
        self.pretrained_base = pretrained_base
        self.preprocess = preprocess_input
        self.pool = GlobalAvgPool2D()
        self.d1 = Dense(256, activation = tf.nn.relu)
        self.final_layer = Dense(9, activation = tf.nn.softmax)

    def call(self, x):
        x = self.rescale(x)
        x = self.preprocess(x)
        x = self.pretrained_base(x)
        x = self.pool(x)
        x = self.d1(x)
        x = self.final_layer(x)
        return x

model = myModel()
num_epochs = 20
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
acc_func_training = tf.keras.metrics.SparseCategoricalAccuracy()
acc_func_val = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss_value = loss_fn(y, pred)
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_func_training.update_state(y, pred)
    return loss_value

@tf.function
def val_step(x_val, y_val):
    pred_val = model(x_val, training=False)
    val_loss_value = loss_fn(y_val, pred_val)
    acc_func_val.update_state(y_val, pred_val)
    return val_loss_value

'''
print('-'*7,'Training started!','-'*7,'\n')
for epoch in range(num_epochs):
    for batch_idx, (x,y) in enumerate(train_dataset):
        loss_value = train_step(x,y)
    train_acc = acc_func_training.result()


    for batch_idx_val, (x_val, y_val) in enumerate(val_dataset):
        val_loss_value = val_step(x_val, y_val)
    val_acc = acc_func_val.result()

    acc_func_val.reset_state()
    acc_func_training.reset_state()

    print(f'Epoch {epoch+1}/{num_epochs}\n'
          f'Training loss: {loss_value}\n'
          f'Training accuracy: {train_acc}\n'
          f'Validation loss: {val_loss_value}\n'
          f'Validation accuracy: {val_acc}\n'
          f'-------------------------------------------------------\n')

'''

callback = tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience = 3)
model.compile(loss = loss_fn,
              optimizer = optimizer,
              metrics = ['accuracy'])

model.fit(train_dataset,
          validation_data = val_dataset,
          epochs = 10,
          callbacks = [callback])

