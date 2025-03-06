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
import os
import random
from torch.utils.data import Dataset

class TripletLoader(Dataset):
    def __init__(self,rootDir,transform=None,triplet=True):
        self.rootDir = rootDir
        self.transform = transform
        self.triplet = triplet
        '''
        classFolders = {}
        for label in os.listdir(rootDir):
            if(os.path.isdir(os.path.join(rootDir, label))):
                classFolders[label] = os.listdir(os.path.join(rootDir, label))
        '''
        #dictionary to correlate label to folder
        self.classFolders = {label: os.listdir(os.path.join(rootDir, label))
                              for label in os.listdir(rootDir) 
                             if os.path.isdir(os.path.join(rootDir, label))} 
        self.classes = list(self.classFolders.keys())
        # (data,label) pairs
        self.allFiles = [(file, label) for label in self.classes for file in self.classFolders[label]]

    def __len__(self):
        return sum(len(files) for files in self.classFolders.values())
        
    def getSingle(self,idx):
        # idx is something the DataSet class handles so that we iterate over the data in an ordered fashion
        file,label = self.allFiles[idx]
        audioPath = os.path.join(self.rootDir, label, file)
        waveform, sr = torchaudio.load(audioPath)

        sample = {
            'waveform': waveform,
            'label': label,
            'sample_rate': sr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getTriplet(self):
        anchorClass = random.choice(self.classes)
        posClass = anchorClass
        negClass = random.choice([c for c in  self.classes if c!=posClass])
        anchorPath,positivePath = random.sample(self.classFolders[posClass],2)
        negativePath = random.choice(self.classFolders[negClass])

        anchor,sr = torchaudio.load(os.path.join(self.rootDir,anchorClass,anchorPath))
        positive,_ = torchaudio.load(os.path.join(self.rootDir,posClass,positivePath))
        negative,_ = torchaudio.load(os.path.join(self.rootDir,negClass,negativePath))

        sample = {
            'anchor': anchor,
            'a':anchorClass,
            'positive': positive,
            'p':posClass,
            'negative': negative,
            'n':negClass,
            'sample_rate': sr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def __getitem__(self, idx):
        if self.triplet:
            return self.getTriplet()
        else:
            return self.getSingle(idx)

def test(rootDir = "./data/SpeechCommands/speech_commands_v0.02"):
    dataset = TripletLoader(rootDir, triplet=True)
    
    sample = dataset[0]
    print("Anchor shape:", sample['anchor'].shape)
    print("Anchor Class:", sample['a'])
    print("Positive shape:", sample['positive'].shape)
    print("Positive Class:", sample['p'])
    print("Negative shape:", sample['negative'].shape)
    print("Negative Class:", sample['n'])
    print("Sample Rate:", sample['sample_rate'])
    
    dataset2 = TripletLoader(rootDir, triplet=False)
    sample = dataset2[0]
    print("Waveform shape:", sample['waveform'].shape)
    print("Label:", sample['label'])
    print("Sample Rate:", sample['sample_rate'])

def Speech(batchSize=4):
  training_set = torchaudio.datasets.SPEECHCOMMANDS("data",download=True,subset='training')
  test_set = torchaudio.datasets.SPEECHCOMMANDS("data",download=True,subset='testing')

  trainLoader = DataLoader(training_set,batch_size=batchSize,shuffle=True)
  testLoader = DataLoader(test_set,batch_size=batchSize,shuffle=False)

  print("DoneSpeech")
  #infoSpeech(trainLoader,testLoader)
  return trainLoader,testLoader
