'''
Context
Lists estimates of the percentage of body fat determined by underwater
weighing and various body circumference measurements for 252 men.

Educational use of the dataset
This data set can be used to illustrate multiple regression techniques.
Accurate measurement of body fat is inconvenient/costly and it is desirable
to have easy methods of estimating body fat that are not inconvenient/costly.

Content
The variables listed below, from left to right, are:

Density determined from underwater weighing
Percent body fat from Siri's (1956) equation
Age (years)
Weight (lbs)
Height (inches)
Neck circumference (cm)
Chest circumference (cm)
Abdomen 2 circumference (cm)
Hip circumference (cm)
Thigh circumference (cm)
Knee circumference (cm)
Ankle circumference (cm)
Biceps (extended) circumference (cm)
Forearm circumference (cm)
Wrist circumference (cm)
(Measurement standards are apparently those listed in Benhke and Wilmore (1974),
pp. 45-48 where, for instance, the abdomen 2 circumference is measured "laterally,
at the level of the iliac crests, and anteriorly, at the umbilicus".)

These data are used to produce the predictive equations for lean body weight
given in the abstract "Generalized body composition prediction equation for men
using simple measurement techniques",
K.W. Penrose, A.G. Nelson, A.G. Fisher, FACSM, Human Performance Research Center,
Brigham Young University, Provo, Utah 84602 as listed in Medicine and Science in Sports and Exercise,
vol. 17, no. 2, April 1985, p. 189.
(The predictive equation were obtained from the first 143 of the 252 cases that are listed below).

More details
A variety of popular health books suggest that the readers assess their health,
at least in part, by estimating their percentage of body fat.
In Bailey (1994), for instance, the reader can estimate body fat from tables
using their age and various skin-fold measurements obtained by using a caliper.
Other texts give predictive equations for body fat using body circumference measurements
(e.g. abdominal circumference) and/or skin-fold measurements.
See, for instance, Behnke and Wilmore (1974), pp. 66-67; Wilmore (1976), p. 247;
or Katch and McArdle (1977), pp. 120-132).

The percentage of body fat for an individual can be estimated once body density has been determined.
Folks (e.g. Siri (1956)) assume that the body consists
of two components - lean body tissue and fat tissue. Letting:

D = Body Density (gm/cm^3)
A = proportion of lean body tissue
B = proportion of fat tissue (A+B=1)
a = density of lean body tissue (gm/cm^3)
b = density of fat tissue (gm/cm^3)
we have:

D = 1/[(A/a) + (B/b)]

solving for B we find:

B = (1/D)*[ab/(a-b)] - [b/(a-b)].

Using the estimates a=1.10 gm/cm^3 and b=0.90 gm/cm^3 (see Katch and McArdle (1977),
p. 111 or Wilmore (1976), p. 123) we come up with "Siri's equation":

Percentage of Body Fat (i.e. 100*B) = 495/D - 450.

Volume, and hence body density, can be accurately measured a variety of ways.
The technique of underwater weighing "computes body volume as the difference
between body weight measured in air and weight measured during water submersion.
In other words, body volume is equal to the loss of weight in
water with the appropriate temperature correction for the water's density"
(Katch and McArdle (1977), p. 113). Using this technique,

Body Density = WA/[(WA-WW)/c.f. - LV]

where:

WA = Weight in air (kg)
WW = Weight in water (kg)
c.f. = Water correction factor (=1 at 39.2 deg F as one-gram of water occupies
exactly one cm^3 at this temperature, =.997 at 76-78 deg F)
LV = Residual Lung Volume (liters)
(Katch and McArdle (1977), p. 115). Other methods of determining body volume
are given in Behnke and Wilmore (1974), p. 22 ff.

Source
The data were generously supplied by Dr. A. Garth Fisher who gave permission
to freely distribute the data and use for non-commercial purposes.

Roger W. Johnson
Department of Mathematics & Computer Science
South Dakota School of Mines & Technology
501 East St. Joseph Street
Rapid City, SD 57701

email address: rwjohnso@silver.sdsmt.edu
web address: http://silver.sdsmt.edu/~rwjohnso

References
Bailey, Covert (1994). Smart Exercise: Burning Fat, Getting Fit, Houghton-Mifflin Co., Boston, pp. 179-186.

Behnke, A.R. and Wilmore, J.H. (1974). Evaluation and Regulation of Body Build and Composition, Prentice-Hall, Englewood Cliffs, N.J.

Siri, W.E. (1956), "Gross composition of the body", in Advances in Biological and Medical Physics, vol. IV, edited by J.H. Lawrence and C.A. Tobias, Academic Press, Inc., New York.

Katch, Frank and McArdle, William (1977). Nutrition, Weight Control, and Exercise, Houghton Mifflin Co., Boston.

Wilmore, Jack (1976). Athletic Training and Physical Fitness: Physiological Principles of the Conditioning Process, Allyn and Bacon, Inc., Boston.

https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/body fat dataset/bodyfat.csv')
y = df.pop('BodyFat')

X_train, X_test, y_train, y_test = train_test_split(df, y)

m1 = LinearRegression()
m1.fit(X_train,y_train)
p1 = m1.predict(X_test)
mse1 = mean_squared_error(y_test, p1)
r1 = r2_score(y_test, p1)

m2 = DecisionTreeRegressor()
m2.fit(X_train,y_train)
p2 = m2.predict(X_test)
mse2 = mean_squared_error(y_test, p2)
r2 = r2_score(y_test, p2)

m3 = RandomForestRegressor()
m3.fit(X_train,y_train)
p3 = m3.predict(X_test)
mse3 = mean_squared_error(y_test, p3)
r3 = r2_score(y_test, p3)

m4 = LinearRegression()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
m4.fit(X_train,y_train)
p4 = m4.predict(X_test)
mse4 = mean_squared_error(y_test, p4)
r4 = r2_score(y_test, p4)

m5 = KNeighborsRegressor()
m5.fit(X_train,y_train)
p5 = m5.predict(X_test)
mse5 = mean_squared_error(y_test, p5)
r5 = r2_score(y_test, p5)

print(f'Linear model\'s mean squared error is {round(mse1,3)}, its r2 is {round(r1,3)}\n')
print(f'Decision tree\'s mean squared error is {round(mse2,3)}, its r2 is {round(r2,3)}\n')
print(f'Random forest\'s mean squared error is {round(mse3,3)}, its r2 is {round(r3,3)}\n')
print(f'Linear model\'s mean squared error (with scaling) is {round(mse4,3)}, its r2 is {round(r4,3)}\n')
print(f'KNN\'s mean squared error (with scaling) is {round(mse5,3)}, its r2 is {round(r5,3)}\n')

'''
A FEW CONSIDERATIONS:
Scaling in this case doesn\'t seem to impact on the accuracy of the linear regressor
'''

