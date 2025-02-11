import torch
import torch.nn as nn
import torch.nn.functional as F

#prediction models
class fashionModel1(nn.Module):
  def __init__(self):
    super().__init__()              #1x28x28
    self.conv1 = nn.Conv2d(1,6,5)   #6x24x24
    self.pool1 = nn.AvgPool2d(2,2)  #6x12x12
    self.conv2 = nn.Conv2d(6,16,3)  #16x8x8
    self.dense1= nn.Linear(16*8*8,512)
    self.dense2= nn.Linear(512,128)
    self.dense3= nn.Linear(128,10)
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

class fashionModel2(nn.Module):
  def __init__(self):
    super().__init__()             #1x28x28
    self.conv1 = nn.Conv2d(1,3,3)  #6x26x26
    self.pool1 = nn.AvgPool2d(2,2) #3x13x13
    self.conv2 = nn.Conv2d(3,6,3)  #6x11x11 -> 6x5x5
    self.conv3 = nn.Conv2d(6,10,3) #10x3x3
    self.dense1= nn.Linear(10*3*3,100) 
    self.dense2= nn.Linear(100,10)
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x=self.pool1(F.relu(self.conv3(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=self.dense2(x)
    return x

class speechModel1(nn.Module):
  def __init__(self):
    super().__init__()              #1x28x28
    self.conv1 = nn.Conv1d(1,6,5)   #6x24x24
    self.pool1 = nn.AvgPool1d(2,2)  #6x12x12
    self.conv2 = nn.Conv1d(6,16,3)  #16x8x8
    self.dense1= nn.Linear(16*8*8,512)
    self.dense2= nn.Linear(512,128)
    self.dense3= nn.Linear(128,10)
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

class speechModel2(nn.Module):
  def __init__(self):
    super().__init__()             #1x28x28
    self.conv1 = nn.Conv1d(1,3,3)  #6x26x26
    self.pool1 = nn.AvgPool1d(2,2) #3x13x13
    self.conv2 = nn.Conv1d(3,6,3)  #6x11x11 -> 6x5x5
    self.conv3 = nn.Conv1d(6,10,3) #10x3x3
    self.dense1= nn.Linear(10*3*3,100) 
    self.dense2= nn.Linear(100,10)
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x=self.pool1(F.relu(self.conv3(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=self.dense2(x)
    return x

def fashionModel1():
  model =fashionModel1()
  return model

def fashionModel2():
  model =fashionModel2()
  return model

def speechModel1():
  model = speechModel1()
  return model

def speechModel2():
  model = speechModel2()
  return model

#representation models
#common loss functions for these include Triplet Loss, Contrastive Loss

class fashionEmbedded(nn.Module):
  def __init__(self):
    super().__init__()              #1x28x28
    self.conv1 = nn.Conv2d(1,6,5)   #6x24x24
    self.pool1 = nn.AvgPool2d(2,2)  #6x12x12
    self.conv2 = nn.Conv2d(6,16,3)  #16x8x8
    self.dense1= nn.Linear(16*8*8,512)
    self.dense2= nn.Linear(512,128)
    self.dense3= nn.Linear(128,32) ##32 dimensional output
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

class speechEmbedded(nn.Module):
  def __init__(self):
    super().__init__()              #1x28x28
    self.conv1 = nn.Conv1d(1,6,5)   #6x24x24
    self.pool1 = nn.AvgPool1d(2,2)  #6x12x12
    self.conv2 = nn.Conv1d(6,16,3)  #16x8x8
    self.dense1= nn.Linear(16*8*8,512)
    self.dense2= nn.Linear(512,128)
    self.dense3= nn.Linear(128,64) #64 dimensional output
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

class smallSpeechPrediction(nn.Module):
  # model for prediction from embedded model in 64 dimensions
