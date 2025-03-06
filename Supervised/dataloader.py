import torch
import torchvision
import torchaudio
import torchvision.transforms as transforms
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def Fashion(mean=0.5,variance=0.5,batchSize=4):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((mean,),(variance,))]
    )
    training_set = torchvision.datasets.FashionMNIST("data",download=False,transform=transform,train=True)
    test_set = torchvision.datasets.FashionMNIST("data",download=False,transform=transform,train=False)
    
    trainLoader = DataLoader(training_set,batch_size = batchSize,shuffle=True)
    testLoader = DataLoader(test_set,batch_size=batchSize,shuffle=False)
    
    infoFashion(trainLoader,testLoader)
    fashionClasses = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    return trainLoader,testLoader,fashionClasses

def Speech(batchSize=4):
    training_set = torchaudio.datasets.SPEECHCOMMANDS("data",download=False,subset='training')
    test_set = torchaudio.datasets.SPEECHCOMMANDS("data",download=False,subset='testing')
    
    trainLoader = DataLoader(training_set,batch_size=batchSize,shuffle=True)
    testLoader = DataLoader(test_set,batch_size=batchSize,shuffle=False)
    speechClasses = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
 'wow', 'yes', 'zero']

  #infoSpeech(trainLoader,testLoader)
    return trainLoader,testLoader,speechClasses

import matplotlib.pyplot as plt

def infoFashion(trainLoader, testLoader):
    print(f"Total Training Batches: {len(trainLoader)}")
    print(f"Total Testing Batches: {len(testLoader)}")

    # Fetch a batch of data
    data_iter = iter(trainLoader)
    images, labels = next(data_iter)

    print(f"Batch Size: {images.shape}")  # Expected (batch_size, 1, 28, 28)

def infoSpeech(speechTrainLoader,speechTestLoader):
  # Print SpeechCommands dataset info
  print(f"Total Speech Training Batches: {len(speechTrainLoader)}")
  print(f"Total Speech Testing Batches: {len(speechTestLoader)}")

  # Fetch a batch of speech data
  speech_iter = iter(speechTrainLoader)
  waveform, sample_rate, label = next(speech_iter)

  print(f"Waveform Shape: {waveform.shape}")  # Shape: (batch_size, 1, num_samples)
  print(f"Sample Rate: {sample_rate}")
  print(f"Labels: {label}")
