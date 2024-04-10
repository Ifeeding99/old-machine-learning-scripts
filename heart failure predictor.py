'''
Context
Cardiovascular diseases (CVDs) are the number 1 cause of death globally,
taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide.
Four out of 5CVD deaths are due to heart attacks and strokes,
and one-third of these deaths occur prematurely in people under 70 years of age.
Heart failure is a common event caused by CVDs and this dataset contains 11 features
that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk
(due to the presence of one or more risk factors such as hypertension,
diabetes, hyperlipidaemia or already established disease) need early detection
and management wherein a machine learning model can be of great help.

Attribute Information
Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]

https://www.kaggle.com/fedesoriano/heart-failure-prediction
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/heart failure/heart.csv')
target = df.pop('HeartDisease')
trasformers = [
    ['sex vectorizer', OneHotEncoder(), [1]],
    ['chest pain vectorizer', OneHotEncoder(), [2]],
    ['resting ecg vectorizer', OneHotEncoder(), [6]],
    ['exercise angina vactorizer', OneHotEncoder(), [8]],
    ['st slope vectorizer', OneHotEncoder(),[10]]
]
ct = ColumnTransformer(trasformers, remainder='passthrough')
df = ct.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(df, target, random_state=0)

model1 = DecisionTreeClassifier(random_state=0)
model1.fit(X_train, y_train)
p1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, p1)

model2 = KNeighborsClassifier()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model2.fit(X_train, y_train)
p2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, p2)

model3 = RandomForestClassifier(random_state=0)
model3.fit(X_train, y_train)
p3 = model3.predict(X_test)
acc3 = accuracy_score(y_test, p3)

model4 = SVC(random_state=0)
model4.fit(X_train, y_train)
p4 = model4.predict(X_test)
acc4 = accuracy_score(y_test, p4)

print(f'decision tree accuracy: {round(acc1*100,3)} % \n '
      f'KNN accuracy: {round(acc2*100,3)} %\n'
      f'Random Forest\'s accuracy: {round(acc3*100,3)} %\n'
      f'SVC\'s accuracy: {round(acc4*100,3)} %')


