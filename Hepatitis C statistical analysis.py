import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/hepatitis C dataset/HepatitisCdata.csv')
d_df = df[(df.Category == '1=Hepatitis')|(df.Category == '2=Fibrosis')|(df.Category =='3=Cirrhosis')]

#sns.scatterplot(x = 'Age', y = 'CHOL', hue = 'Sex', data = df)
#sns.histplot(data=d_df, hue = 'Sex',x='Sex')   # graph showing the amounts of ill men and ill women
sns.histplot(data=df,x='Sex',hue='Sex')     # graph that show the total men and women involved

def counter(d,s,v,disease=''): # d is the dataframe, s is the column, v is the value to search for
    c = 0
    if disease == '':
        for el in d[s]:
            if el == v:
                c += 1
    else:
        d2 = d[(d[s]==v)&(d['Category']==disease)] # here I check for males and females affected by a specific disease
        for el in d2[s]:
            if el == v:
                c += 1

    return c

# counting
males_tot = counter(df,'Sex','m')
females_tot = counter(df,'Sex','f')
males_ill = counter(d_df,'Sex','m')
females_ill = counter(d_df,'Sex','f')
males_hepatitis = counter(d_df,'Sex','m',disease='1=Hepatitis')
males_fibrosis = counter(d_df,'Sex','m',disease='2=Fibrosis')
males_cirrhosis = counter(d_df,'Sex','m',disease='3=Cirrhosis')
females_hepatitis = counter(d_df,'Sex','f',disease='1=Hepatitis')
females_fibrosis = counter(d_df,'Sex','f',disease='2=Fibrosis')
females_cirrhosis = counter(d_df,'Sex','f',disease='3=Cirrhosis')

print(f'total males: {males_tot}, total females: {females_tot}\n'
      f'ill males: {males_ill} ({round(males_ill/males_tot*100,3)} %)\n'
      f'ill females: {females_ill} ({round(females_ill/females_tot*100,3)} %)\n'
      f'males with hepatitis C: {males_hepatitis}, they are {round(males_hepatitis/males_ill*100,3)} % of male diseased and {round(males_hepatitis/males_tot,3)} % of the males total\n'
      f'females with hepatitis C: {females_hepatitis}, they are {round(females_hepatitis/females_ill*100,3)} % of female diseased and {round(females_hepatitis/females_tot,3)} % of the females total\n'
      f'males with fibrosis: {males_fibrosis}, they are {round(males_fibrosis/males_ill*100,3)} % of male diseased and {round(males_fibrosis/males_tot,3)} % of the males total\n'
      f'females with fibrosis: {females_fibrosis}, they are {round(females_fibrosis/females_ill*100,3)} % of male diseased and {round(females_fibrosis/females_tot,3)} % of the females total\n'
      f'males with cirrhosis: {males_cirrhosis}, they are {round(males_cirrhosis/males_ill*100,3)} % of male diseased and {round(males_cirrhosis/males_tot,3)} % of the males total\n'
      f'females with cirrhosis: {females_cirrhosis}, they are {round(females_cirrhosis/females_ill*100,3)} % of male diseased and {round(females_cirrhosis/females_tot,3)} % of the females total\n')



plt.show()