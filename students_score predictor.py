'''
This beginner level data set has 403 rows and 6 columns.
It is a real dataset about the students' knowledge status about the subject of Electrical DC Machines.

STG: The degree of study time for goal object materials
SCG: The degree of repetition number of user for goal object materials
STR: The degree of study time of user for related objects with goal object
LPR: The exam performance of user for related objects with goal object
PEG: The exam performance of user for goal objects
UNS: The knowledge level of user (Very Low, Low, Middle, High)

https://www.kaggle.com/datasets/farkhod77/predict-students-level
'''
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


df = pd.read_excel('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/students_score/Predict_student_ knowledge_level.xls')
y = df.pop(' UNS')

encoder = LabelEncoder()
y = encoder.fit_transform(y)

trfs = [ # transoformers
    ['Normalizer', StandardScaler(), [0,1,2,3,4]]
]
ct = ColumnTransformer(trfs, remainder='passthrough')
df = pd.DataFrame(ct.fit_transform(df))

X_train, X_test, y_train, y_test = train_test_split(df, y, random_state = 0)

# models
'''
m1 = XGBClassifier(random_state = 0)
# search space
search_space = {
    'n_estimators' : [50 + 10*i for i in range (50)],
    'gamma' : [0.01, 0.1],
    'learning_rate' : [(i+1)/1000 for i in range (20)]
}
GS = GridSearchCV(estimator = m1,
                  param_grid = search_space,
                  scoring = ['accuracy'],
                  refit = 'accuracy',
                  cv = 5,
                  verbose = 4)
GS.fit(X_train, y_train)
print(GS.best_params_)
# {'gamma': 0.01, 'learning_rate': 0.009, 'n_estimators': 500} # best parameters
'''
m1 = XGBClassifier(random_state = 0, learning_rate = 0.009, n_estimators = 500, gamma = 0.01)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)
acc1 = accuracy_score(y_test, p1)


m2 = SVC()
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)
acc2 = accuracy_score(y_test, p2)


m3 = RandomForestClassifier(random_state = 0, n_estimators=200)
'''
search_space = {
    'n_estimators' : [50+10*i for i in range(30)]
}
GS = GridSearchCV(m3,
                  param_grid = search_space,
                  refit = True,
                  cv = 5,
                  verbose = 4,
                  scoring = ['accuracy']
    )
GS.fit(X_train, y_train)
print(GS.best_params_)
# {'n_estimators': 200}
'''
m3.fit(X_train, y_train)
p3 = m3.predict(X_test)
acc3 = accuracy_score(y_test, p3)

print(f'xgboost\'s asccuracy: {round(acc1*100,3)} % ')
print(f'SVM\'s asccuracy: {round(acc2*100,3)} % ')
print(f'Random forest\'s asccuracy: {round(acc3*100,3)} % ')