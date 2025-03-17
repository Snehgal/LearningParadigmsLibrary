import torch
from torch.autograd.function import _SingleLevelFunction
import torch.nn as nn
import torch.nn.functional as F

#prediction models
class fashionModel1(nn.Module):
  def __init__(self):
    super(fashionModel1,self).__init__()              #1x28x28
    self.conv1 = nn.Conv2d(1,6,5)   #6x24x24
    self.pool1 = nn.AvgPool2d(2,2)  #6x12x12
    self.conv2 = nn.Conv2d(6,16,3)  #16x5x5
    self.dense1 = nn.Linear(16*5*5,512)  # Change from 16*8*8 to 16*5*5
    self.dense2= nn.Linear(512,128)
    self.dense3= nn.Linear(128,10)
  
  def forward(self,x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool1(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.dense1(x))
    x = F.relu(self.dense2(x))
    x = self.dense3(x)
    return x


class fashionModel2(nn.Module):
  def __init__(self):
    super(fashionModel2,self).__init__()             #1x28x28
    self.conv1 = nn.Conv2d(1,3,3)  #6x26x26
    self.pool1 = nn.AvgPool2d(2,2) #3x13x13
    self.conv2 = nn.Conv2d(3,6,3)  #6x11x11 -> 6x5x5
    self.conv3 = nn.Conv2d(6,10,3) #10x3x3
    self.dense1= nn.Linear(10*3*3,100) 
    self.dense2= nn.Linear(100,10)
  
  def forward(self, x):  
    x = self.pool1(F.relu(self.conv1(x)))  
    x = F.relu(self.conv2(x))
    x = self.pool1(x)  
    x = F.relu(self.conv3(x))
    x = torch.flatten(x, 1)
    x = F.relu(self.dense1(x))
    x = self.dense2(x)
    return x

class speechModel1(nn.Module):
    def __init__(self):
        super(speechModel1, self).__init__()              # 1x16k
        self.conv1 = nn.Conv1d(1, 6, 7, 2)  # 6x7997
        self.pool1 = nn.AvgPool1d(2, 2)    # 6x3998
        self.conv2 = nn.Conv1d(6, 16, 3)   # 16x3994 -> 16x1997 ->16x998
        self.dense1 = nn.Linear(15984 , 1024)  # FIXED
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, 32)    # 64-dimensional output

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(x)
        print(f"Before flattening: {x.shape}")  # Debugging shape
        x = torch.flatten(x, 1)
        print(f"After flattening: {x.shape}")  # Debugging shape
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x



class speechModel2(nn.Module):
  def __init__(self):
    super(speechModel2,self).__init__()           #1x16k
    self.conv1 = nn.Conv1d(1,6,7,2) #6x7997
    self.pool1 = nn.AvgPool1d(2,2)  #6x3998
    self.conv2 = nn.Conv1d(6,16,5)  #16x3994 -> 16x1997
    self.conv3 = nn.Conv1d(16,32,3) #32x1995 ->32x997
    self.dense1= nn.Linear(32*997,1024)
    self.dense2= nn.Linear(1024,256)
    self.dense3= nn.Linear(256,32)
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x=self.pool1(F.relu(self.conv3(x)))
    print(f"Before flattening: {x.shape}")
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

#representation models
#common loss functions for these include Triplet Loss, Contrastive Loss

class fashionEmbedded(nn.Module):
  def __init__(self):
    super(fashionEmbedded,self).__init__()              #1x28x28
    self.conv1 = nn.Conv2d(1,6,5)   #6x24x24
    self.pool1 = nn.AvgPool2d(2,2)  #6x12x12
    self.conv2 = nn.Conv2d(6,16,3)  #16x8x8
    self.dense1= nn.Linear(16*8*8,512)
    self.dense2= nn.Linear(512,256)
    self.dense3= nn.Linear(256,64) #64 dimensional output
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

class smallFashionPrediction(nn.Module):
  def __init__(self):
    super(smallFashionPrediction,self).__init__()
    self.dense1= nn.Linear(64,32)
    self.dense2= nn.Linear(32,10) #64 dimensional output

  def forward(self,x):
      x=F.relu(self.dense1(x))
      x=self.dense2(x)
      return x

class speechEmbedded(nn.Module):
  def __init__(self):
    super(speechEmbedded,self).__init__()              #1x16k
    self.conv1 = nn.Conv1d(1,6,7,2)   #6x7997
    self.pool1 = nn.AvgPool1d(2,2)  #6x3998
    self.conv2 = nn.Conv1d(6,16,3)  #16x3994 -> 16x1997 ->16x998
    self.dense1= nn.Linear(16*998,1024)
    self.dense2= nn.Linear(1024,256)
    self.dense3= nn.Linear(256,100) #64 dimensional output
  
  def forward(self,x):
    x=self.pool1(F.relu(self.conv1(x)))
    x=self.pool1(F.relu(self.conv2(x)))
    x=self.pool1(x)
    x = torch.flatten(x,1)
    x=F.relu(self.dense1(x))
    x=F.relu(self.dense2(x))
    x=self.dense3(x)
    return x

class smallSpeechPrediction(nn.Module):
  # model for prediction from embedded model in 64 dimensions
  def __init__(self):
    super(smallSpeechPrediction,self).__init__()
    self.dense0 = nn.Linear(100,32)
  
  def forward(self,x):
    x = self.dense0(x)
    return x
