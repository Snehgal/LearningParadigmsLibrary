import dataloader as dl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

'''
loading FashionMNIST and printing 4 images
'''
def imshow(img,mean=0.5,variance=0.5):
    img = img*variance + mean     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
batchSize = 4
fashionTrain,fashionTest,fashionClasses = dl.Fashion(batchSize=batchSize)
#model=model.fashionModel1()

# runs in collab directly only
dataiter = iter(fashionTrain)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{fashionClasses[labels[j]]:5s}' for j in range(batchSize)))

'''
loading SpeechAudio
'''
speechTrain,speechTest,speechClasses = dl.Speech()


def trainPrediction(trainLoader,testLoader,optimizer,model,criterion,epochs = 10):
    '''
    train Prediction models
    '''
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
    '''
    train Embedded models
    '''
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
