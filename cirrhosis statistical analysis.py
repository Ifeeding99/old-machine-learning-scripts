import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/cirrhosis dataset/cirrhosis.csv')
# removing useless columns
df.pop('Drug')
df.pop('ID')
df.pop('Status')
df.pop('N_Days')
# removing patients with only basic checks
df = df[0:312]

fig, ax = plt.subplots(3,2)
sns.violinplot(x='Stage', y='Age', hue = 'Sex', data = df, ax = ax[0,0])
sns.histplot(df.Sex, ax = ax[0,1])
sns.scatterplot(x='Age',y='Cholesterol',hue='Sex', data = df, ax = ax[1,0])
sns.kdeplot(x='Age', hue='Sex',data = df, ax = ax[1,1])
sns.kdeplot(x='Stage', hue='Sex', shade = True, data = df, ax = ax[2,0])
sns.regplot(x='Age', y = 'Albumin', data = df, ax=ax[2,1])
plt.tight_layout()
plt.show()