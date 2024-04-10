'''
This code examines statistical aspects of the stellar objects dataset

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/stellar classification dataset/star_classification.csv')
df_stars = df[df['class'] == 'STAR']
df_galaxies = df[df['class'] == 'GALAXY']
df_quasars = df[df['class'] == 'QSO']


#sns.scatterplot(x='i', y='redshift',hue='class',data = df)
#sns.kdeplot(x='redshift',hue='class',data=df, shade = True) # unscaled distribution of redshift
sns.violinplot(x='class',y='redshift',hue='class',data= df) # this visualizes the distribution of the redshift
#sns.histplot(x='class',hue='class',data = df) # this visualizes how many galaxies, stars and quasars are there
plt.grid()
plt.show()

