'''
Context
In astronomy, stellar classification is the classification of stars
based on their spectral characteristics. The classification scheme of galaxies,
quasars, and stars is one of the most fundamental in astronomy.
The early cataloguing of stars and their distribution in the sky has led
to the understanding that they make up our own galaxy and,
following the distinction that Andromeda was a separate galaxy to our own,
numerous galaxies began to be surveyed as more powerful telescopes were built.
This datasat aims to classificate stars, galaxies, and quasars based on their spectral characteristics.

Content
The data consists of 100,000 observations of space
taken by the SDSS (Sloan Digital Sky Survey). Every observation is described
by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.

obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
alpha = Right Ascension angle (at J2000 epoch)
delta = Declination angle (at J2000 epoch)
u = Ultraviolet filter in the photometric system
g = Green filter in the photometric system
r = Red filter in the photometric system
i = Near Infrared filter in the photometric system
z = Infrared filter in the photometric system
run_ID = Run Number used to identify the specific scan
rereun_ID = Rerun Number to specify how the image was processed
cam_col = Camera column to identify the scanline within the run
field_ID = Field number to identify each field
spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
class = object class (galaxy, star or quasar object)
redshift = redshift value based on the increase in wavelength
plate = plate ID, identifies each plate in SDSS
MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation

https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

'''

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# preprocessing data
df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/stellar classification dataset/star_classification.csv')
target = df.pop('class')

cols = df.columns
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df)) # the scaler does not return a dataframe, but a numpy array
df.columns = cols # reassigning labels to columns
label_transformer = LabelEncoder()
target = pd.DataFrame(label_transformer.fit_transform(target))
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25)

model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
p1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, p1)
print('model1 done')

model2 = RandomForestClassifier()
y_train = np.array(y_train).ravel()
model2.fit(X_train, y_train)
p1 = model1.predict(X_test)
acc2 = accuracy_score(y_test, p1)
print('model 2 done')

print(f'Decision tree\'s accuracy is {round(acc1*100,3)} % \n'
      f'Random Forest\'s accuracy is {round(acc2*100,3)} %')
print(f'execution time: {round(time.process_time(),3)} s')

'''
A FEW CONSIDERATIONS:
K nearest neighbours is REALLY slow and inaccurate(86%/87%) in this scenario.
Random forest takes a lot of time and doesn't do much better than a decision tree, which is way faster
'''


