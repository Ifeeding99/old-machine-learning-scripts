from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dataset = load_iris()
X = dataset['data']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
p = model.predict(X_test)
acc = accuracy_score(y_test,p)
print(f'L\'accuratezza è: {acc*100}%')
