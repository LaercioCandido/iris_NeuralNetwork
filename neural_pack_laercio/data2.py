import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import random

path = './iris.csv'
classes = {"Iris-setosa": 0, "Iris-versicolor": 1,
    "Iris-virginica":2}

class DatasetCSV(Dataset):
    def __init__(self, path, label_transform):
        self.df = pd.read_csv(path)
        self.labeltransform = label_transform
    def __len__(self):
        return len(self.df)  ## tem que retornar o número de linhas! 
    def __getitem__(self, idx):
        x = self.df.loc[idx, :'petalwidth']
        y = self.df.loc[idx,'class']
        x = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float32)
        y = self.labeltransform.transform(y)
        y = torch.tensor(np.asarray(y, dtype=int))
        return (x, y)

class LabelTransform():
    def __init__(self, classes):
        self.classes = classes
    def transform(self, label):
        label = self.classes[label]
        return label

class LoadData():
    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = list(range(len(self.dataset)))

    def prepare_data(self):
        ##  shuffle all the indexes and splits the numbers into train, cv and test
        random.shuffle(self.idxs)
        train_idx = int(len(self.dataset) * 0.6)
        cv_idx = train_idx + int(len(self.dataset) * 0.2)
        self.train = self.idxs[: train_idx]
        self.cv = self.idxs[train_idx: cv_idx]
        self.test = self.idxs[cv_idx:]

    def train_loader(self):
        train_dataset = Subset(self.dataset, self.train)
        trainloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
        print("size of training dataset:", len(train_dataset))
        return (trainloader)

    def cv_loader(self):
        cv_dataset = Subset(self.dataset, self.cv)
        cvloader = DataLoader(cv_dataset, batch_size=12, shuffle=True)
        print("size of cross-validation dataset:", len(cv_dataset))
        return (cvloader)
    
    def test_loader(self):
        test_dataset = Subset(self.dataset, self.test)
        testloader = DataLoader(test_dataset, batch_size=12, shuffle=True)
        print("size of testing dataset:", len(test_dataset))
        return (testloader)