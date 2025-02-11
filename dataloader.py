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

def Fashion(mean=0.5,median=0.5,batchSize=4):
  transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((mean,),(median,))]
  )
  training_set = torchvision.datasets.FashionMNIST("data",download=True,transform=transform,train=True)
  test_set = torchvision.datasets.FashionMNIST("data",download=True,transform=transform,train=False)

  trainLoader = DataLoader(training_set,batch_size = batchSize,shuffle=True)
  testLoader = DataLoader(test_set,batch_size=batchSize,shuffle=False)

  infoFasion(trainLoader,testLoader)
  return trainLoader,testLoader

def Speech(batchSize=4):
  training_set = torchaudio.datasets.SPEECHCOMMANDS("data",download=True,subset='training')
  test_set = torchaudio.datasets.SPEECHCOMMANDS("data",download=True,subset='testing')

  trainLoader = DataLoader(training_set,batch_size=batchSize,shuffle=True)
  testLoader = DataLoader(test_set,batch_size=batchSize,shuffle=False)

  infoSpeech(trainLoader,testLoader)
  return trainLoader,testLoader

def show_images(images, labels):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 3))
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.show()

def infoFasion(trainLoader,testLoader):
  #using GPT
  print(f"Total Training Batches: {len(trainLoader)}")
  print(f"Total Testing Batches: {len(testLoader)}")

  # Fetch a batch of data
  data_iter = iter(trainLoader)
  images, labels = next(data_iter)

  print(f"Batch Size: {images.shape}")  # Shape: (batch_size, 1, 28, 28)
  show_images(images[:5], labels[:5])

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
