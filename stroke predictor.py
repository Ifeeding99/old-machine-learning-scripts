'''
Context
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally,
responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke
based on the input parameters like gender, age, various diseases, and smoking status.
Each row in the data provides relavant information about the patient.

Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

'''

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/stroke dataset/healthcare-dataset-stroke-data.csv')
df.pop('id')
target = df.pop('stroke')

transformers = [
    ['gender vectorizer', OneHotEncoder(), [0]],
    ['marriage vactorizer', OneHotEncoder(), [4]],
    ['work-type vectorizer', OneHotEncoder(), [5]],
    ['residence-type vectorizer', OneHotEncoder(), [6]],
    ['bmi nan remover', SimpleImputer(), [8]],
    ['smoking vectorizer', OneHotEncoder(),[9]]

]

ct = ColumnTransformer(transformers, remainder = 'passthrough')
df = ct.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(df, target)

model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
p1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, p1)

model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
p2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, p2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model3 = KNeighborsClassifier()
model3.fit(X_train, y_train)
p3 = model3.predict(X_test)
acc3 = accuracy_score(y_test, p3)

print(f'Decision tree accuracy: {round(acc1*100,3)} % \n',
      f'Random forest accuracy: {round(acc2*100,3)} % \n',
      f'KNN accuracy: {round(acc3*100,3)} %')



