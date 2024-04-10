'''
Attribute information:

```
sick, negative. | classes

age: continuous.
sex: M, F.
on thyroxine: f, t.
query on thyroxine: f, t.
on antithyroid medication: f, t.
sick: f, t.
pregnant: f, t.
thyroid surgery: f, t.
I131 treatment: f, t.
query hypothyroid: f, t.
query hyperthyroid: f, t.
lithium: f, t.
goitre: f, t.
tumor: f, t.
hypopituitary: f, t.
psych: f, t.
TSH measured: f, t.
TSH: continuous.
T3 measured: f, t.
T3: continuous.
TT4 measured: f, t.
TT4: continuous.
T4U measured: f, t.
T4U: continuous.
FTI measured: f, t.
FTI: continuous.
TBG measured: f, t.
TBG: continuous.
referral source: WEST, STMW, SVHC, SVI, SVHD, other.
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/thyroid sickness dataset/dataset_thyroid_sick.csv')
y = df.pop('Class')
df.pop('referral_source') # non predictive information
df.pop('TSH_measured')
df.pop('T3_measured')
df.pop('TT4_measured')
df.pop('T4U_measured')
df.pop('FTI_measured')
df.pop('TBG_measured')
df.pop('TBG') # empty column
l_encoder = LabelEncoder()
y = l_encoder.fit_transform(y) # encoding the target

for col in df.columns:
    df[col].replace(to_replace='?',value=np.nan,inplace=True) # I had to replace '?' with NaN in order to apply the imputer

#print(df.columns[df.isnull().any()].tolist()) # prints the columns with nans

transformers = [
    ['age imputer', SimpleImputer(), [0]],
    ['sex imputer', SimpleImputer(strategy='most_frequent'), [1]],
    ['on_thyroxine encoder', OrdinalEncoder(), [2]],
    ['query_on_thyroxine encoder', OrdinalEncoder(), [3]],
    ['on_antithyroid_medication', OrdinalEncoder(), [4]],
    ['sick encoder', OrdinalEncoder(), [5]],
    ['pregnant encoder', OrdinalEncoder(), [6]],
    ['thyroid_surgery encoder', OrdinalEncoder(), [7]],
    ['I131_treatment encoder', OrdinalEncoder(), [8]],
    ['query_hypothyroid encoder', OrdinalEncoder(), [9]],
    ['query_hyperthyroid encoder', OrdinalEncoder(), [10]],
    ['lithium encoder', OrdinalEncoder(), [11]],
    ['goitre encoder', OrdinalEncoder(), [12]],
    ['tumor encoder', OrdinalEncoder(), [13]],
    ['hypopituitary encoder', OrdinalEncoder(), [14]],
    ['psych encoder', OrdinalEncoder(), [15]],
    ['TSH imputer', SimpleImputer(), [16]],
    ['T3 imputer', SimpleImputer(), [17]],
    ['TT4 imputer', SimpleImputer(), [18]],
    ['T4U imputer', SimpleImputer(),[19]],
    ['FTI imputer', SimpleImputer(), [20]]
]
col = df.columns
ct1 = ColumnTransformer(transformers, remainder='passthrough')
df = pd.DataFrame(ct1.fit_transform(df))
df.columns = col
col = df.columns

sex_encoder = OrdinalEncoder()
df['sex'] = sex_encoder.fit_transform(np.array(df['sex']).reshape(-1,1)) # encodes the sex variable (m,f) => (0,1)

scalers = [
    ['age scaler', MinMaxScaler(), [0]],
    ['TSH scaler', MinMaxScaler(), [16]],
    ['T3 scaler', MinMaxScaler(), [17]],
    ['TT4 scaler', MinMaxScaler(), [18]],
    ['T4U scaler', MinMaxScaler(), [19]],
    ['FTI scaler', MinMaxScaler(), [20]]
]
ct3 = ColumnTransformer(scalers, remainder = 'passthrough')
df = pd.DataFrame(ct3.fit_transform(df))
df.columns = col

X_train, X_test, y_train, y_test = train_test_split(df,y)

m1 = RandomForestClassifier()
m1.fit(X_train, y_train)
p2 = m1.predict(X_test)
acc1 = accuracy_score(y_test, p2)

print(f'Random Forest\'s accuracy: {round(acc1*100,3)} %')
