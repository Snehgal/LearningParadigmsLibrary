import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import dataloader as dl
import lossFunction as lf
import model as m
import optimizer as opt

# Select device (GPU if available, otherwise CPU)
device = "cpu"
print("Starting...")
# Define dataset path and speech commands classes
rootDir = "./data/SpeechCommands/speech_commands_v0.02"
speechClasses = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
 'wow', 'yes', 'zero']

# Load dataset
dataset = dl.TripletLoader(rootDir, transform=None, triplet=True)
print("Data")

# Define loss function
lossFn = lf.TripletLoss(margin=0.2)
print("LossFn")

# Initialize model and move to device
model = m.ResNet18(numClasses=len(speechClasses), inChannels=1).to(device)
# model = m.ResNet50(numClasses=len(speechClasses), inChannels=1).to(device)
# model = m.ResNet101(numClasses=len(speechClasses), inChannels=1).to(device)
print("Model")

# Define optimizer and scheduler
optimizer = opt.AdamW(model)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)
print("OptSched")

# Training settings
batchSize = 4
numEpochs = 5
trainLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
print("Loader")

# Training Loop
for epoch in range(numEpochs):
    print(epoch)
    model.train()  # Set model to training mode
    epochLoss = 0.0

    for batch in trainLoader:
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)

        # Forward pass
        anchorOut = model(anchor)
        positiveOut = model(positive)
        negativeOut = model(negative)

        # Compute loss
        lossValue = lossFn(anchorOut, positiveOut, negativeOut)

        # Backward pass
        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()

        # Accumulate loss
        epochLoss += lossValue.item()

    # Step scheduler
    scheduler.step()

    # Print epoch loss
    print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {epochLoss/len(trainLoader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

print("Training Completed.")
