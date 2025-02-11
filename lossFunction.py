import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#classical loss functions
def MSE():
  loss = nn.MSELoss()
  return loss

def CrossEntropyLoss():
  loss = nn.CrossEntropyLoss()
  return loss

def BCELoss():
  loss = nn.BCELoss()
  return loss

def TripletMarginLoss():
  loss = nn.TripletMarginLoss()
  return loss

#trainable loss functions
#GPT helped because I didnt understand a lot

class TrainableTripleLoss(nn.Module):
  def __init__(self):
    super(TrainableTripleLoss,self).__init__()
    self.margin = nn.Parameter(torch.tensor(1.0))

  def forward(self,anchor,positive,negative):
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
    return loss.mean()

class LearnedLoss(nn.Module):
  def __init__(self):
    super(LearnedLoss,self).__init__()
    self.dense = nn.Linear(2,1)

  def forward(self, pred, target):
        error = torch.abs(pred - target)
        loss = self.fc(torch.stack([pred, error], dim=1))
        return loss.mean()
