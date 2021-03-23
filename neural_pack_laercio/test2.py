#!/usr/bin/env python
# coding: utf-8

# In[3]:


# from benchmark import datasets
# from benchmark.datasets import fetch
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import random
from data import DatasetCSV, LabelTransform, LoadData


# # In[4]:


# datasets = ['iris']
# # fetch(datasets, file_type='csv', overwrite=False)
# path = './iris.csv'
# classes = {"Iris-setosa": 0, "Iris-versicolor": 1,
#     "Iris-virginica":2}


# # In[5]:


# classes['Iris-setosa']


# # In[6]:


# class DatasetCSV(Dataset):
#     def __init__(self, path, label_transform):
#         self.df = pd.read_csv(path)
#         self.labeltransform = label_transform
#     def __len__(self):
#         return len(self.df)  ## tem que retornar o número de linhas! 
#     def __getitem__(self, idx):
#         x = self.df.loc[idx, :'petalwidth']
#         y = self.df.loc[idx,'class']
#         x = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float32)
#         y = self.labeltransform.transform(y)
#         y = torch.tensor(np.asarray(y, dtype=int))
#         return (x, y)


# # In[7]:


# class LabelTransform():
#     def __init__(self, classes):
#         self.classes = classes
#     def transform(self, label):
#         label = self.classes[label]
#         return label


# In[10]:
path = './iris.csv'
classes = {"Iris-setosa": 0, "Iris-versicolor": 1,
    "Iris-virginica":2}

labeltransform = LabelTransform(classes)
dataset_return = DatasetCSV(path, labeltransform)
load_data_obj = LoadData(dataset_return)

load_data_obj.prepare_data()


print()
# idxs = list(range(len(dataset_return)))
# random.shuffle(idxs)
# train_idx = int(len(dataset_return) * 0.6)
# cv_idx = train_idx + int(len(dataset_return) * 0.2)
# train = idxs[: train_idx]
# cv = idxs[train_idx: cv_idx]
# test = idxs[cv_idx:]
# # In[ ]:

# train_dataset = Subset(dataset_return, train)
# cv_dataset = Subset(dataset_return, cv)
# test_dataset = Subset(dataset_return, test)

# trainloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
# cvloader = DataLoader(cv_dataset, batch_size=12, shuffle=True)
# testloader = DataLoader(test_dataset, batch_size=12, shuffle=True)
# print("size of training dataset:", len(train_dataset))
# print("size of cross-validation dataset:", len(cv_dataset))
# print("size of testing dataset:", len(test_dataset))
# print(train_dataset[0])


class Net(nn.Module):
    
    def __init__(self, features, num_inner_neurons, output_size):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(features, num_inner_neurons)
        self.fc2 = nn.Linear(num_inner_neurons, num_inner_neurons)
        self.out = nn.Linear(num_inner_neurons, output_size)
    
    def forward(self, x_train):
        x_train = F.normalize(x_train)
        x_train = F.relu(self.fc1(x_train))
        x_train = F.relu(self.fc2(x_train))
        y_predict = self.out(x_train)
        
        return y_predict
# In[ ]:

num_inner_neurons = 100
output_size = 3
features = 4
net = Net(features, num_inner_neurons, output_size)

# x_tensor = torch.from_numpy(x_train).float()
# y_tensor = torch.from_numpy(y_train).float()


# # In[ ]:


optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
trainloader = load_data_obj.train_loader()

for epoch in range(10):
    for batch_x, batch_y in trainloader:
        optimizer.zero_grad()
        prediction = net(batch_x)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
        print("Monitor loss", loss)