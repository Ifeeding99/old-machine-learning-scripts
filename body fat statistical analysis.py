import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/body fat dataset/bodyfat.csv')
#print(df.columns)
cols = df.columns
transfromers = [
    ['scaler', MinMaxScaler(), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
]
ct = ColumnTransformer(transfromers, remainder = 'passthrough')
df = pd.DataFrame(ct.fit_transform(df))
df.columns = cols
fig, ax = plt.subplots(2,2)
sns.regplot(x='BodyFat',y='Density', data = df, ax = ax[0,0])
sns.regplot(x='BodyFat',y='Weight', data = df, ax = ax[0,1], color = 'orange')
sns.regplot(x='BodyFat',y='Height', data = df, ax = ax[1,0], color = 'red')
sns.regplot(x='BodyFat',y='Abdomen', data = df, ax = ax[1,1], color = 'green')
plt.title('Body fat statistical analysis')
plt.tight_layout()
plt.show()