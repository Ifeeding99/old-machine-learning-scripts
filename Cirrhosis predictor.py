'''
Context
Cirrhosis is a late stage of scarring (fibrosis) of the liver caused
by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism.
The following data contains the information collected from the Mayo Clinic trial
in primary biliary cirrhosis (PBC) of the liver conducted between 1974 and 1984.
A description of the clinical background for the trial and the covariates recorded here
is in Chapter 0, especially Section 0.2 of Fleming and Harrington, Counting
Processes and Survival Analysis, Wiley, 1991.
A more extended discussion can be found in Dickson, et al., Hepatology 10:1-7 (1989)
and in Markus, et al., N Eng J of Med 320:1709-13 (1989).

A total of 424 PBC patients, referred to Mayo Clinic during that ten-year interval,
met eligibility criteria for the randomized placebo-controlled trial of the drug D-penicillamine.
The first 312 cases in the dataset participated in the randomized trial and contain largely complete data.
The additional 112 cases did not participate in the clinical trial but consented
to have basic measurements recorded and to be followed for survival.
Six of those cases were lost to follow-up shortly after diagnosis,
so the data here are on an additional 106 cases as well as the 312 randomized participants.

Attribute Information
1) ID: unique identifier
2) N_Days: number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986
3) Status: status of the patient C (censored), CL (censored due to liver tx), or D (death)
4) Drug: type of drug D-penicillamine or placebo
5) Age: age in [days]
6) Sex: M (male) or F (female)
7) Ascites: presence of ascites N (No) or Y (Yes)
8) Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes)
9) Spiders: presence of spiders N (No) or Y (Yes)
10) Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)
11) Bilirubin: serum bilirubin in [mg/dl]
12) Cholesterol: serum cholesterol in [mg/dl]
13) Albumin: albumin in [gm/dl]
14) Copper: urine copper in [ug/day]
15) Alk_Phos: alkaline phosphatase in [U/liter]
16) SGOT: SGOT in [U/ml]
17) Triglycerides: triglicerides in [mg/dl]
18) Platelets: platelets per cubic [ml/1000]
19) Prothrombin: prothrombin time in seconds [s]
20) Stage: histologic stage of disease (1, 2, 3, or 4)

https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset

I will ignore the columns 'Drug','Status','ID', 'N_Days' and I will try to predict the stage of the desease ('Stage' column)
You can predict many charaterics of this dataset, predicting the stage is particularly challeging:
I couldn't get to more than 55% accuracy more or less (often less)
I then tried to predict wheter the cirrhosis was severe (stage 3 or 4) or at its beginning (stage 1 or 2).
The accuracy in the latter case is around 80% at best but more often between 65% and 75%
'''
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

seed = np.random.seed(42)

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/cirrhosis dataset/cirrhosis.csv')
# removing useless columns
df.pop('Drug')
df.pop('ID')
df.pop('Status')
df.pop('N_Days')
# removing patients with only basic checks
df = df[0:312]
y = df.pop('Stage')
#print(df)

# in the following lines I reduce the 4 categories of Stage to 2 ((1 or 2) = 0, (3 or 4) = 1)
y = list(y)
'''for i,n in enumerate(y):
    if n >=3:
        y[i] = 1
    else:
        y[i] = 0'''
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# getting columns with nan values in them
nan_val = df.isna()
nan_col = nan_val.any()
df_nan_cols = df.columns[nan_col].tolist()
#print(df_nan_cols)
# the columns with nan values are 'Cholesterol', 'Copper', 'Tryglicerides' and 'Platelets'

transformers = [
    ['encoder', OneHotEncoder(), [1,2,3,4,5]],
    ['scaler', StandardScaler(), [0,6,7,8,9,10,11,12,13,14]]
]
ct = ColumnTransformer(transformers, remainder = 'passthrough')
df = pd.DataFrame(ct.fit_transform(df))

nan_val = df.isna()
nan_col = nan_val.any()
df_nan_cols = df.columns[nan_col].tolist()

impute_col = [
    ['imputer', SimpleImputer(), df_nan_cols]
]
c_imputer = ColumnTransformer(impute_col, remainder = 'passthrough')
df = pd.DataFrame(c_imputer.fit_transform(df))
pca = PCA(n_components=8)
df = pca.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(df,y)

m1 = DecisionTreeClassifier()
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)
acc1 = accuracy_score(y_test, p1)

m2 = LogisticRegression(max_iter = 10000) # increase the max_iterations to get better accuracy
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)
acc2 = accuracy_score(y_test, p2)

m3 = RandomForestClassifier()
m3.fit(X_train, y_train)
p3 = m3.predict(X_test)
acc3 = accuracy_score(y_test, p3)

m4 = KNeighborsClassifier()
m4.fit(X_train, y_train)
p4 = m4.predict(X_test)
acc4 = accuracy_score(y_test, p4)

m5 = BernoulliNB()
m5.fit(X_train, y_train)
p5 = m5.predict(X_test)
acc5 = accuracy_score(y_test, p5)

m6 = SVC()
m6.fit(X_train, y_train)
p6 = m6.predict(X_test)
acc6 = accuracy_score(y_test, p6)

m7 = XGBClassifier(n_estimators = 10000, learning_rate = 0.005)
m7.fit(X_train, y_train, eval_set = [(X_test, y_test)], early_stopping_rounds = 10, verbose = False)
p7 = m7.predict(X_test)
acc7 = accuracy_score(y_test, p7)


m8 = MLPClassifier(hidden_layer_sizes=(20,20))
m8.fit(X_train, y_train)
p8 = m8.predict(X_test)
acc8 = accuracy_score(y_test, p8)

print(f'Decision tree\'s accuracy: {round(acc1*100,3)} % \n'
      f'Logistic classifier\'s accuracy: {round(acc2*100,3)} % \n'
      f'Random forest\'s accuracy: {round(acc3*100,3)} % \n'
      f'KNN\'s accuracy: {round(acc4*100,3)} % \n'
      f'Naive Bayes\'s accuracy: {round(acc5*100,3)} % \n'
      f'Support vector classifier\'s accuracy: {round(acc6*100,3)} %\n'
      f'XGBoost\'s accuracy: {round(acc7*100,3)} %\n'
      f'Neural network\'s accuracy: {round(acc8*100,3)} %')


