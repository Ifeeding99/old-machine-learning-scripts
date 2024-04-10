'''
Context
The data set contains laboratory values of blood donors
and Hepatitis C patients and demographic values like age.
The data was obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/HCV+data
https://www.kaggle.com/fedesoriano/hepatitis-c-dataset

Content
All attributes except Category and Sex are numerical.
Attributes 1 to 4 refer to the data of the patient:
1) X (Patient ID/No.)
2) Category (diagnosis) (values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis')
3) Age (in years)
4) Sex (f,m)
Attributes 5 to 14 refer to laboratory data:
5) ALB      Albumin Blood Test
6) ALP      Alkaline phosphatase
7) ALT      Alanine Transaminase
8) AST      Aspartate Transaminase
9) BIL      Bilirubin
10) CHE     Acetylcholinesterase
11) CHOL    Cholesterol
12) CREA    Creatinine
13) GGT     Gamma-Glutamyl Transferase
14) PROT    Proteins

The target attribute for classification is Category (2):
blood donors vs. Hepatitis C patients
(including its progress ('just' Hepatitis C, Fibrosis, Cirrhosis).

Acknowledgements
Creators: Ralf Lichtinghagen, Frank Klawonn, Georg Hoffmann
Donor: Ralf Lichtinghagen: Institute of Clinical Chemistry; Medical University Hannover (MHH); Hannover, Germany; lichtinghagen.ralf '@' mh-hannover.de
Donor: Frank Klawonn; Helmholtz Centre for Infection Research; Braunschweig, Germany; frank.klawonn '@' helmholtz-hzi.de
Donor: Georg Hoffmann; Trillium GmbH; Grafrath, Germany; georg.hoffmann '@' trillium.de

Relevant Papers
Lichtinghagen R et al. J Hepatol 2013; 59: 236-42
Hoffmann G et al. Using machine learning techniques to generate laboratory diagnostic pathways â€“ a case study. J Lab Precis Med 2018; 3: 58-67

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/hepatitis C dataset/HepatitisCdata.csv')
df = df.iloc[:,1:]  # here I remove the first useless column, in which there are the numbers of the various cases
y = df.pop('Category')  # isolating target variable
df_copy = df.copy() # this line gets a copy of the dataframe

transformers = [
    ['sex vectorizer', OneHotEncoder(), [1]],
    ['nan remover', SimpleImputer(), [0,2,3,4,5,6,7,8,9,10,11]]
]
ct = ColumnTransformer(transformers, remainder = 'passthrough')
df = ct.fit_transform(df)
scaler = MinMaxScaler()
df_f = pd.DataFrame(scaler.fit_transform(df)) # the scaler and the imputer do not return a dataframe, but a numpy array
#df.columns = df_copy.columns # this line reassign the name to the corresponding columns
X_train, X_test, y_train, y_test = train_test_split(df_f, y)

model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
p1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, p1)

model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
p2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, p2)

model3 = KNeighborsClassifier()
model3.fit(X_train, y_train)
p3 = model3.predict(X_test)
acc3 = accuracy_score(y_test, p3)

print(f'Decision tree\'s accuracy: {round(acc1*100,3)}')
print(f'Random forest\'s accuracy: {round(acc2*100,3)}')
print(f'KNN\'s accuracy: {round(acc3*100,3)}')



