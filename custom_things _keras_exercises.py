import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LayerNormalization, Dense, Flatten, Normalization
from tensorflow.keras.datasets.fashion_mnist import load_data


class MyLayerNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        s = input_shape[-1]
        self.a = tf.Variable(tf.ones(s,), trainable=True, dtype = tf.float32)
        self.b = tf.Variable(tf.zeros(s,), trainable = True, dtype = tf.float32)

    def call(self,inputs):
        epsilon = tf.constant(1e-7, dtype = tf.float32)
        mean, var = tf.nn.moments(inputs, axes = -1, keepdims=True)
        std = tf.sqrt(var)
        transformed_x = self.a * (inputs-mean)/(std + epsilon) + self.b
        return transformed_x

'''
l = MyLayerNorm()
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]], dtype = tf.float32)
b = l(a)
print(b)

l_default = LayerNormalization()
b_2 = l_default(a)
print(b_2)
'''

(X_train, y_train),(X_test,y_test) = load_data()
norm = Normalization()
norm.adapt(X_train)

upper_layers = Sequential()
for i in range(3):
    upper_layers.add(Dense(16,activation='selu'))
    upper_layers.add(Dense(10, activation=tf.nn.softmax))

lower_layers= Sequential()
for i in range(3):
    lower_layers.add(Dense(16,activation='selu'))


input_ = Input(shape=(28,28))
x = norm(input_)
x = Flatten()(x)
x = lower_layers(x)
x = upper_layers(x)
my_model = Model(inputs = input_, outputs = x)

opt_1 = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov = True)
opt_2 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
train_acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()
train_loss_mean = tf.keras.metrics.Mean()
val_loss_mean = tf.keras.metrics.Mean()


batch_size = 512
train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

n_epochs = 50
for epoch in range(n_epochs):
    for batch_idx,data in enumerate(train_dataset):
        X,y = data
        with tf.GradientTape(persistent=True) as tape: # persistent is False by default, so
            pred = my_model(X, training = True)
            loss = loss_fn(y,pred)

        grad1 = tape.gradient(loss, upper_layers.trainable_variables)
        opt_1.apply_gradients(zip(grad1, upper_layers.trainable_variables))

        grad2 = tape.gradient(loss, lower_layers.trainable_variables)
        opt_2.apply_gradients(zip(grad2, lower_layers.trainable_variables))

        del tape # if you keep the tape (you set it to persistent) you have to delete it manually

        train_acc_fn.update_state(y,pred)
        train_loss_mean.update_state(loss)

    train_acc = train_acc_fn.result().numpy()
    train_acc_fn.reset_state()
    train_loss = train_loss_mean.result().numpy()
    train_loss_mean.reset_state()

    for batch_idx, data in enumerate(val_dataset):
        X,y = data
        pred = my_model(X, training = False)
        val_loss =  loss_fn(y, pred)
        val_acc_fn.update_state(y,pred)
        val_loss_mean.update_state(val_loss)

    val_acc = val_acc_fn.result().numpy()
    val_acc_fn.reset_state()
    val_loss = val_loss_mean.result().numpy()
    val_loss_mean.reset_state()


    print(f'epoch {epoch+1}/{n_epochs+1}, train loss: {train_loss:.3f} train acc: {train_acc:.3f}'
          f' val loss: {val_loss:.3f} val acc: {val_acc:.3f}')








