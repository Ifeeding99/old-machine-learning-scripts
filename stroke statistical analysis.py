'''
This is a statistical analysis of the dataset used in 'stroke predictor', see that if you want more context.
'''
import pandas as pd

df = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/stroke dataset/healthcare-dataset-stroke-data.csv')
df.pop('id')
strokes = df[df.stroke == 1]
male_strokes = df[(df.gender == 'Male') & (df.stroke == 1)]
female_strokes = df[(df.gender == 'Female') & (df.stroke == 1)]
other_strokes = df[(df.gender == 'Other') & (df.stroke == 1)]
tot_female = df[df.gender == 'Female']
tot_male = df[df.gender == 'Male']
tot_other = df[df.gender == 'Other']

print(f'total cases: {df.shape[0]}\n\n'
      f'number of females: {tot_female.shape[0]},  female cases: {female_strokes.shape[0]}, incidence in females: {round(female_strokes.shape[0]/tot_female.shape[0]*100,3)} %\n'
      f'number of males: {tot_male.shape[0]},  male cases: {male_strokes.shape[0]}, incidence in males: {round(male_strokes.shape[0]/tot_male.shape[0]*100,3)} %\n'
      f'number of others: {tot_other.shape[0]}, cases in others: {other_strokes.shape[0]}, incidence in others: {round(other_strokes.shape[0]/tot_other.shape[0]*100,3)} %\n')

df_is_married = df[df.ever_married=='Yes']
df_not_married = df[df.ever_married=='No']
df_MS = df[(df.ever_married=='Yes')& (df.stroke==1)]
df_NMS = df[(df.ever_married == 'No') & (df.stroke == 1)]

tot_married = df_is_married.shape[0]
tot_not_married = df_not_married.shape[0]
married_stroke = df_MS.shape[0]
not_married_stroke = df_NMS.shape[0]
print(f'number of people married: {tot_married}, total strokes between them: {married_stroke}, rate of incidence is {round(married_stroke/tot_not_married*100,3)} %\n'
      f'number of people not married: {tot_not_married}, total strokes between them: {not_married_stroke}, rate of incidence is {round(not_married_stroke/tot_not_married*100,3)} %\n')


tot_rural_citizens = df[df.Residence_type == 'Rural']
tot_urban_citizens = df[df.Residence_type == 'Urban']
stroke_rural = df[(df.Residence_type == 'Rural')&(df.stroke == 1)]
stroke_urban = df[(df.Residence_type == 'Urban')&(df.stroke == 1)]
print(f'number of people living in rural regions: {tot_rural_citizens.shape[0]}, number of strokes between them: {stroke_rural.shape[0]}, rate of incidende: {round(stroke_rural.shape[0]/tot_rural_citizens.shape[0]*100,3)} %\n'
      f'number of people living in urban regions: {tot_urban_citizens.shape[0]}, number of strokes between them: {stroke_urban.shape[0]}, rate of incidende: {round(stroke_urban.shape[0]/tot_urban_citizens.shape[0]*100,3)} %\n')


tot_children_workers = df[df.work_type == 'children'] # they are children, they are NOT adults that work with children
tot_public_workers = df[df.work_type == 'Govt_jov']
tot_private_workers = df[df.work_type == 'Private']
tot_self_employed = df[df.work_type == 'Self-employed']
tot_non_workers = df[df.work_type == 'Never_worked']
children_strokes = df[(df.work_type=='children')&(df.stroke == 1)]
government_strokes = df[(df.work_type == 'Govt_jov')&(df.stroke == 1)]
private_strokes = df[(df.work_type == 'Private')&(df.stroke == 1)]
self_employed_strokes = df[(df.work_type == 'Self-employed')&(df.stroke == 1)]
non_workers_strokes = df[(df.work_type == 'Never_worked')&(df.stroke == 1)]
print(f'number children is {tot_children_workers.shape[0]}, the number of strokes between them is {children_strokes.shape[0]}, the rate of incidence is {round(children_strokes.shape[0]/tot_children_workers.shape[0]*100,3)} %\n'
      f'number of people working for the government is {tot_public_workers.shape[0]}, the number of strokes between them is {government_strokes.shape[0]}, the rate of incidence is {round(0,3)} %\n'
      f'number of people working in private is {tot_private_workers.shape[0]}, the number of strokes between them is {private_strokes.shape[0]}, the rate of incidence is {round(private_strokes.shape[0]/tot_private_workers.shape[0]*100,3)} %\n'
      f'number of people self-employed is {tot_self_employed.shape[0]}, the number of strokes between them is {self_employed_strokes.shape[0]}, the rate of incidence is {round(self_employed_strokes.shape[0]/tot_self_employed.shape[0]*100,3)} %\n'
      f'number of non-working people is {tot_non_workers.shape[0]}, the number of strokes between them is {non_workers_strokes.shape[0]}, the rate of incidence is {round(non_workers_strokes.shape[0]/tot_non_workers.shape[0]*100,3)} %\n')

'''
CONCLUSIONS:
In this dataset, males seem more likely to have a stroke than females, but it appears the opposite is true,
this may be caused by the fact that there are almost a thousand more female entries in the dataset than males entries.
It seems that being married increases the risk of stroke by a bit, 
although the example of non-married people are a half of the example of married people.
Living in a city or outside it doesn't seem to have a great effect on the incidence rate,
urban citizen are more likely to get a stroke by a tiny bit according to the data.
Talking about jobs, the ones at higher risk seem to be the self-employed people.
'''
