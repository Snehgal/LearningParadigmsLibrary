import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def trainPrediction(trainLoader,testLoader,optimizer,model,criterion,epochs = 10):
  runningLoss = 0.0
  for epoch in range(epochs):
    for i,data in  enumerate(trainLoader,0):
      inputs,labels = data
      optimizer.zero_grad()
      outputs = model(inputs) #forward pass
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
          running_loss = 0.0
  print('Finished Training')

def trainEmbedded(trainLoader,testLoader,optimizer,model,criterion,epochs = 10):
  running_loss = 0
  for epoch in range(epochs):
    for i,data in  enumerate(trainLoader,0):
      anchor,positive,negative = data
      optimizer.zero_grad()

      anchorOut = model(anchor) #forward pass
      positiveOut = model(positive)
      negativeOut = model(negative)
      loss = criterion(ancherOut,positiveOut,negativeOut)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
          running_loss = 0.0
  print('Finished Training')


