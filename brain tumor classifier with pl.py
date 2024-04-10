import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer

training_path = 'C:/Users/User01/Desktop/brain tumors/Training'
validation_path = 'C:/Users/User01/Desktop/brain tumors/Testing'

img_size = 150
def rescale(img):
    return img/255

class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (3,3), stride = 1, padding = 'valid')
        self.conv2 = torch.nn.Conv2d(32, 64, (3,3), stride = 1, padding='valid')
        self.pool = torch.nn.MaxPool2d((2,2))
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(82944, 150)
        self.linear2 = torch.nn.Linear(150,4)
        self.acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = 4)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        pred = self(images)
        loss = F.cross_entropy(pred, labels)
        self.acc(pred, labels)
        self.log('accuracy: ', self.acc, on_step = True, prog_bar=True)
        return {'loss':loss}

    def train_dataloader(self):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            #torchvision.transforms.RandomRotation(60),
            #torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            rescale
        ])
        train_dataset = torchvision.datasets.ImageFolder(training_path, transforms)
        loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 4)
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-3)



if __name__ == '__main__':
    trainer = Trainer(fast_dev_run=False, max_epochs=100)
    model = NeuralNetwork()
    trainer.fit(model)