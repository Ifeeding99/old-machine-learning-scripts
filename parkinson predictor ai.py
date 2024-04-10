
'''
Parkinson is a nasty desease

Context
Parkinson’s Disease (PD) is a degenerative neurological disorder
marked by decreased dopamine levels in the brain.
It manifests itself through a deterioration of movement,
including the presence of tremors and stiffness.
There is commonly a marked effect on speech, including dysarthria (difficulty articulating sounds),
hypophonia (lowered volume), and monotone (reduced pitch range).
Additionally, cognitive impairments and changes in mood can occur, and risk of
dementia is increased.

Traditional diagnosis of Parkinson’s Disease involves a clinician
taking a neurological history of the patient and observing motor skills
in various situations. Since there is no definitive laboratory test to diagnose PD,
diagnosis is often difficult, particularly in the early stages when motor effects are not yet severe.
Monitoring progression of the disease over time requires repeated clinic visits by the patient.
An effective screening process, particularly one that doesn’t require a clinic visit, would be beneficial.
Since PD patients exhibit characteristic vocal features,
voice recordings are a useful and non-invasive tool for diagnosis.
If machine learning algorithms could be applied to a voice recording dataset to accurately diagnosis PD,
this would be an effective screening step prior to an appointment with a clinician.


Attribute Information:
Matrix column entries (attributes):
name - ASCII subject name and recording number
MDVP:Fo(Hz) - Average vocal fundamental frequency
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
MDVP:Flo(Hz) - Minimum vocal fundamental frequency
MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several
measures of variation in fundamental frequency
MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
NHR,HNR - Two measures of ratio of noise to tonal components in the voice
status - Health status of the subject (one) - Parkinson's, (zero) - healthy
RPDE,D2 - Two nonlinear dynamical complexity measures
DFA - Signal fractal scaling exponent
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv('C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/parkinson dataset/Parkinsson disease.csv')
dataset.pop('name')
y = dataset.pop('status')

X_train, X_test, y_train, y_test = train_test_split(dataset, y)
model1 = DecisionTreeRegressor()
model1.fit(X_train, y_train)
p1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, p1)

model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)
p2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, p2)

model3 = KNeighborsClassifier()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model3.fit(X_train, y_train)
p3 = model3.predict(X_test)
acc3 = accuracy_score(y_test, p3)

print(f'accuracy of Decision Tree is {round(acc1*100,3)} %')
print(f'accuracy of KNN without scaling is {round(acc2*100,3)} %')
print(f'accuracy of KNN with scaling is {round(acc3*100,3)} %')