from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = load_breast_cancer()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X,y)
scaler = StandardScaler()
X_train = torch.from_numpy((scaler.fit_transform(X_train)).astype(np.float32))
X_test = torch.from_numpy(scaler.transform(X_test).astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

class myModel(torch.nn.Module):
    def __init__(self, input_features, output_features=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted



model = myModel(30)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

def calculate_accuracy(pred, real):
    tot = len(real)
    correct = 0
    for i,el in enumerate(pred):
        if el >= 0.5:
            el = 1
        else:
            el = 0

        if el == real[i]:
            correct += 1
    acc = correct/tot
    return acc


for epoch in range (200):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    accuracy = calculate_accuracy(y_pred,y_train)

    print(f'epoch {epoch}  loss: {round(loss.item(),3)}   accuracy: {round(accuracy,3)}')

print('##############TEST#################')
with torch.no_grad():
    pred = model(X_test)
    acc = calculate_accuracy(pred, y_test)
    print(f'Test accuracy is {acc:.4f}')


