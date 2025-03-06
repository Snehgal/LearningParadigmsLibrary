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
#speechTrain,speechTest = dl.Speech()
