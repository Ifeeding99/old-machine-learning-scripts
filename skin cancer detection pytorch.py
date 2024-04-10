import torch
import torchvision
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


training_path = 'C:/Users/User01/Desktop/skin_cancer/train'
val_path = 'C:/Users/User01/Desktop/skin_cancer/test'
img_size = 224

def Rescale(a):
    return a/255

transforms = Compose([
    ToTensor(),
    Resize((224,224)),
    Rescale
])

train_dataset = ImageFolder(training_path, transform=transforms)
val_dataset = ImageFolder(val_path, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# model
class myModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,16, (3,3), stride = 1, padding='same')
        self.conv2 = torch.nn.Conv2d(16, 64, (3,3), stride = 1, padding = 'same')
        self.pool = torch.nn.MaxPool2d((2,2))
        self.d1 = torch.nn.Linear(200704, 512)
        self.d2 = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        output = self.sigmoid(x)
        return output

model = myModel()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
num_epochs = 10

def calculate_acc(a,b):
    counter = 0
    for i, el in enumerate(a):
        if torch.max(a[i]) == b[i]:
            counter += 1

    acc = counter/a.shape[0]

for epoch in range(num_epochs):
    for batc_idx, (X_batch, y_batch) in enumerate(train_loader):
        p = model(X_batch)
        y_batch = y_batch.view(-1,1)
        y_batch = y_batch.to(torch.float32)
        optimizer.zero_grad()
        loss = criterion(p, y_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for batc_idx, (X_batch_val, y_batch_val) in enumerate(val_loader):
            p_val = model(X_batch_val)
            y_batch_val = y_batch_val.view(-1, 1)
            y_batch_val = y_batch_val.to(torch.float32)
            val_loss = criterion(p_val, y_batch_val)
        #val_acc = calculate_acc()

    print(f'epoch {epoch + 1}  loss = {round(loss.item(), 3)}    val_loss = {round(val_loss.item(), 3)}')