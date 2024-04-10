import numpy as np
import os
import pandas as pd
import torch
import torchvision

torch.set_num_threads(4)

training_path = 'C:/Users/User01/Desktop/brain tumors/Training'
validation_path = 'C:/Users/User01/Desktop/brain tumors/Testing'
img_size = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def rescale (img):
    return img/255

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size,img_size)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(60),
    torchvision.transforms.ToTensor(),
    rescale
])

val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size,img_size)),
    torchvision.transforms.ToTensor(),
    rescale
])

train_dataset = torchvision.datasets.ImageFolder(training_path, transforms)
val_dataset = torchvision.datasets.ImageFolder(validation_path, val_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32)

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64,(3,3), padding='same')
        self.conv2 = torch.nn.Conv2d(64, 128, (3,3), padding='same')
        self.pool = torch.nn.MaxPool2d((2,2))
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(1605632, 128)
        self.linear2 = torch.nn.Linear(128, 4)
        self.myrelu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(100):
    for batch, (X,y) in enumerate(train_loader):
        X.to(device)
        y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(batch)

    with torch.no_grad():
        for val_batch, (X_val, y_val) in enumerate(val_loader):
            pred_val = model(X_val)
            val_loss = criterion(pred_val, y_val)

    print(f'epoch {epoch+1}  loss = {round(loss.item(),3)}    val_loss = {round(val_loss.item(),3)}')