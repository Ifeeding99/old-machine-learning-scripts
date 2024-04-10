'''
For more context, view the 'stroke predictor' code
'''
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

acc_list = []
est_list = []
X_train, X_test, y_train, y_test = train_test_split(df,target)
for i in range(0,10):
    model = XGBClassifier(n_estimators = 300, learning_rate = 0.001*(i+1))
    model.fit(X_train, y_train, eval_set = [(X_test, y_test)], early_stopping_rounds = 5, verbose = False)
    p = model.predict(X_test)
    acc = accuracy_score(y_test, p)
    acc_list.append(acc)
    est_list.append(0.001*(i+1))

plt.plot(est_list, acc_list, marker = '.')
plt.show()
