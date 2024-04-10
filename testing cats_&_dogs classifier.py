'''
https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

model1 = tf.keras.models.load_model('Dogs_&_cats_classifier') # here I load the model I wrote in 'first computer vision script'
model2 = tf.keras.models.load_model('Dogs_&_cats_classifier 2.0') # this is the model with 0.99 accuracy
print('models loaded')
image_path = 'C:/Users/User01/Desktop/dogs-vs-cats/test1/12005.jpg'
#image_path = 'C:/Users/User01/Desktop/dogs-vs-cats/maya.jpeg' # per testare con le foto di Maya


img = cv2.imread(image_path)
cv2.imshow('image', img) # it lets you see the original image, comment this if you want to compete against the machine
cv2.waitKey(0) # waits until a key is pressed


img2 = cv2.resize(img,(64,64)) # image preparation for model 2
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = img_to_array(img2)
img2 = np.expand_dims(img2, axis = 0)


img1 = cv2.resize(img,(50,50)) # the models takes in input rgb images  50x50 pixels
# for model2.0 use 64x64 images
# for the original version use 50x50 images
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
plt.imshow(img1) # it lets you see the rescaled image
plt.show()

img1 = img_to_array(img1)
img1 = np.expand_dims(img1, axis = 0) # it's because of how tf works
# tensorflow wants arrays containing each a single pixel


p1 = int(model1.predict(img1))
print('model 1 prediction')
if p1 == 0:
    print(f'cat (output: {p1})')
else:
    print(f'dog (output: {p1})')

print('\n')
p2 = float(model2.predict(img2))
print('model 2 prediction')
if p2 == 0:
    print(f'cat (output: {p2})')
else:
    print(f'dog (output: {p2})')