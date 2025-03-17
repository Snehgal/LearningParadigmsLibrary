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
    """
    A PyTorch dataset for loading audio data in either single-sample or triplet form.
    
    Attributes:
        rootDir (str): Root directory containing labeled subdirectories of audio files.
        transform (callable, optional): A function/transform to apply to samples.
        triplet (bool): Whether to return triplets (anchor, positive, negative) or single samples.
    """
    def __init__(self, rootDir, transform=None, triplet=True,targetLength = 16000):
        self.rootDir = rootDir
        self.transform = transform
        self.triplet = triplet
        self.targetLength = targetLength
        
        # Dictionary mapping class labels to their respective file lists
        self.classFolders = {label: os.listdir(os.path.join(rootDir, label))
                              for label in os.listdir(rootDir) 
                             if os.path.isdir(os.path.join(rootDir, label))} 
        self.classes = list(self.classFolders.keys())
        
        # List of all (file, label) pairs
        self.allFiles = [(file, label) for label in self.classes for file in self.classFolders[label]]

    def __len__(self):
        """Returns the total number of audio files in the dataset."""
        return sum(len(files) for files in self.classFolders.values())
        
    def getSingle(self, idx):
        """
        Retrieves a single audio sample and its metadata.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the waveform, label, and sample rate.
        """
        file, label = self.allFiles[idx]
        audioPath = os.path.join(self.rootDir, label, file)
        waveform, sr = torchaudio.load(audioPath)

        waveform = pad_or_truncate(waveform)
        
        sample = {
            'waveform': waveform,
            'label': label,
            'sampleRate': sr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getTriplet(self):
        """
        Retrieves a triplet of audio samples: anchor, positive, and negative.
        
        Returns:
            dict: A dictionary containing waveforms and corresponding class labels.
        """
        anchorClass = random.choice(self.classes)
        posClass = anchorClass
        negClass = random.choice([c for c in  self.classes if c != posClass])
        anchorPath, positivePath = random.sample(self.classFolders[posClass], 2)
        negativePath = random.choice(self.classFolders[negClass])

        anchor, sr = torchaudio.load(os.path.join(self.rootDir, anchorClass, anchorPath))
        positive, _ = torchaudio.load(os.path.join(self.rootDir, posClass, positivePath))
        negative, _ = torchaudio.load(os.path.join(self.rootDir, negClass, negativePath))

        # Apply padding/truncation
        anchor = self.pad_or_truncate(anchor)
        positive = self.pad_or_truncate(positive)
        negative = self.pad_or_truncate(negative)
        
        sample = {
            'anchor': anchor,
            'anchorClass': anchorClass,
            'positive': positive,
            'positiveClass': posClass,
            'negative': negative,
            'negativeClass': negClass,
            'sampleRate': sr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset. Returns either a triplet or a single sample.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: A dictionary containing the requested sample(s).
        """
        if self.triplet:
            return self.getTriplet()
        else:
            return self.getSingle(idx)

    # help from GPT
    def pad_or_truncate(self, tensor):
        """Ensure the tensor is exactly self.targetLength samples long."""
        length = tensor.shape[-1]
        if length > self.targetLength:
            return tensor[:, :self.targetLength]  # Truncate
        elif length < self.targetLength:
            pad = torch.zeros((tensor.shape[0], self.targetLength - length))  # Pad
            return torch.cat((tensor, pad), dim=-1)
        return tensor

        
def test(rootDir = "./data/SpeechCommands/speech_commands_v0.02"):
    """
    Tests the TripletLoader by loading a sample and printing its properties.
    
    Args:
        rootDir (str, optional): Path to the dataset root directory.
    """
    dataset = TripletLoader(rootDir, triplet=True)
    
    sample = dataset[0]
    print("Anchor shape:", sample['anchor'].shape)
    print("Anchor Class:", sample['anchorClass'])
    print("Positive shape:", sample['positive'].shape)
    print("Positive Class:", sample['positiveClass'])
    print("Negative shape:", sample['negative'].shape)
    print("Negative Class:", sample['negativeClass'])
    print("Sample Rate:", sample['sampleRate'])
    
    dataset2 = TripletLoader(rootDir, triplet=False)
    sample = dataset2[0]
    print("Waveform shape:", sample['waveform'].shape)
    print("Label:", sample['label'])
    print("Sample Rate:", sample['sampleRate'])

def Speech(batchSize=4):
    """
    Loads the SpeechCommands dataset and creates data loaders for training and testing.
    
    Args:
        batchSize (int, optional): Number of samples per batch. Default is 4.
    
    Returns:
        tuple: Data loaders for training and testing sets.
    """
    training_set = torchaudio.datasets.SPEECHCOMMANDS("data", download=True, subset='training')
    test_set = torchaudio.datasets.SPEECHCOMMANDS("data", download=True, subset='testing')

    trainLoader = DataLoader(training_set, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=batchSize, shuffle=False)

    print("DoneSpeech")
    return trainLoader, testLoader
