Skip to main content
TrainingParadigms.ipynb
TrainingParadigms.ipynb_
Testing Log of negative numbers

[ ]
1 cell hidden
Pipeline

[1]
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#run in case files are deleted
!git init
!git remote add origin https://github.com/ChiragSehga1/TrainingParadigms.git
!git clone https://github.com/ChiragSehga1/TrainingParadigms.git
!cd TrainingParadigms
!mv TrainingParadigms/* .

hint: Using 'master' as the name for the initial branch. This default branch name
hint: is subject to change. To configure the initial branch name to use in all
hint: of your new repositories, which will suppress this warning, call:
hint: 
hint: 	git config --global init.defaultBranch <name>
hint: 
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint: 
hint: 	git branch -m <name>
Initialized empty Git repository in /content/.git/
Cloning into 'TrainingParadigms'...
remote: Enumerating objects: 45, done.
remote: Counting objects: 100% (45/45), done.
remote: Compressing objects: 100% (40/40), done.
remote: Total 45 (delta 17), reused 0 (delta 0), pack-reused 0 (from 0)
Receiving objects: 100% (45/45), 16.12 KiB | 16.12 MiB/s, done.
Resolving deltas: 100% (17/17), done.

[3]
1m
!python3 main.py
Total Training Batches: 15000
Total Testing Batches: 2500
Batch Size: torch.Size([4, 1, 28, 28])
Figure(640x480)
Shirt Coat  Coat  Bag  
100% 2.26G/2.26G [00:21<00:00, 115MB/s]
DoneSpeech

[7]
2s
import dataloader as dl
batchSize = 4
fashionTrain,fashionTest,fashionClasses = dl.Fashion(batchSize=batchSize)
def imshow(img,mean=0.5,variance=0.5):
    img = img*variance + mean     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
dataiter = iter(fashionTrain)
images, labels = next(dataiter)



[ ]
Colab paid products - Cancel contracts here
252627282930313233343536373839404142434445464748495051
class TrainableTripleLoss(nn.Module):
  def __init__(self):
    super(TrainableTripleLoss,self).__init__()
    self.margin = nn.Parameter(torch.tensor(1.0))

  def forward(self,anchor,positive,negative):
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
    return loss.mean()


  0s
completed at 2:43 AM
.weight.data.fill_(1) m.conv2.weight.data.fill_(1)
