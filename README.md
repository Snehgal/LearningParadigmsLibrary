# DeepLearningParadigms Library

A comprehensive PyTorch-based library for implementing both **Supervised** and **Unsupervised** learning paradigms. This library provides modular components for building, training, and evaluating neural networks with support for various datasets, model architectures, loss functions, and optimization techniques.

[Colab Link](https://colab.research.google.com/drive/1enBSP1OgK7ujzIqUXSWuSX7p8Lfmg9Wk#scrollTo=wpzAW0zzLPhl)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Features

### Supervised Learning
- **Pre-built Models**: CNN architectures for FashionMNIST and speech recognition
- **Embedding Models**: Generate fixed-dimensional representations for downstream tasks
- **Loss Functions**: MSE, CrossEntropy, BCE, Triplet Loss (including trainable variants)
- **Optimizers**: Adam, SGD, AdamW with customizable parameters
- **Data Loaders**: FashionMNIST and SpeechCommands dataset support
- **Training Utilities**: Complete training pipelines for both prediction and embedding tasks

### Unsupervised Learning
- **Advanced Architectures**: ResNet18, ResNet34, ResNet50, ResNet101 (1D variants)
- **Triplet Learning**: Custom triplet data loaders and loss functions
- **Clustering**: Trainable clustering loss with learnable centroids
- **Metric Learning**: Sophisticated embedding generation for similarity learning
- **Custom Data Loaders**: Support for custom audio datasets with triplet sampling

## 📦 Installation

Since this is a non-published library, you need to install it manually:

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- torchvision
- torchaudio

### Manual Installation

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/ChiragSehga1/TrainingParadigms.git
   cd TrainingParadigms
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add to Python path** (choose one method):
   
   **Method A: Direct import** (when working in the same directory):
   ```python
   from Supervised import dataloader, model, lossFunction, optimizer
   from Unsupervised import dataloader, model, lossFunction, optimizer
   ```

   **Method B: Add to PYTHONPATH**:
   ```bash
   # Windows
   set PYTHONPATH=%PYTHONPATH%;C:\path\to\TrainingParadigms
   
   # Linux/Mac
   export PYTHONPATH=$PYTHONPATH:/path/to/TrainingParadigms
   ```

   **Method C: Install in development mode**:
   ```bash
   pip install -e .
   ```

## 🎯 Quick Start

### Supervised Learning Example

```python
import torch
from Supervised import dataloader as dl
from Supervised import model, lossFunction, optimizer

# Load FashionMNIST dataset
trainLoader, testLoader, classes = dl.Fashion(batchSize=32)

# Initialize model, loss, and optimizer
net = model.fashionModel1()
criterion = lossFunction.CrossEntropyLoss()
optim = optimizer.Adam(net, lr=0.001)

# Training loop
for epoch in range(10):
    for inputs, labels in trainLoader:
        optim.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
```

### Unsupervised Learning Example

```python
import torch
from Unsupervised import dataloader as dl
from Unsupervised import model, lossFunction, optimizer

# Load triplet dataset
dataset = dl.TripletLoader("./data/SpeechCommands/speech_commands_v0.02", triplet=True)
trainLoader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize embedding model
net = model.ResNet18Embedding(embeddingDim=512)
criterion = lossFunction.TripletLoss(margin=0.2)
optim = optimizer.AdamW(net, lr=0.001)

# Training loop
for epoch in range(5):
    for batch in trainLoader:
        anchor_out = net(batch['anchor'])
        positive_out = net(batch['positive'])
        negative_out = net(batch['negative'])
        
        loss = criterion(anchor_out, positive_out, negative_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
```

## 🎓 Supervised Learning

### Models

#### Prediction Models
- **`fashionModel1`**: Basic CNN for FashionMNIST classification
- **`fashionModel2`**: Alternative CNN architecture
- **`speechModel`**: CNN for speech command recognition

#### Embedding Models
- **`fashionEmbedded`**: Generate 64-dimensional embeddings from FashionMNIST
- **`speechEmbedded`**: Generate 100-dimensional embeddings from speech data
- **`smallFashionPrediction`**: Classifier for fashion embeddings
- **`smallSpeechPrediction`**: Classifier for speech embeddings

### Data Loaders

```python
# FashionMNIST
trainLoader, testLoader, classes = dl.Fashion(
    mean=0.5, 
    variance=0.5, 
    batchSize=32, 
    download=True
)

# Speech Commands
trainLoader, testLoader, classes = dl.Speech(
    batchSize=16, 
    download=True
)
```

### Loss Functions

```python
# Standard losses
criterion = lossFunction.MSE()
criterion = lossFunction.CrossEntropyLoss()
criterion = lossFunction.BCELoss()
criterion = lossFunction.TripletMarginLoss(margin=1.0)

# Trainable losses
criterion = lossFunction.TrainableTripleLoss()
```

### Optimizers

```python
optimizer = optimizer.Adam(model, lr=0.001, betas=(0.9, 0.999))
optimizer = optimizer.SGD(model, lr=0.001, momentum=0.5)
optimizer = optimizer.AdamW(model, lr=0.001)
```

## 🔬 Unsupervised Learning

### Models

#### ResNet Architectures (1D)
- **`ResNet18`**: 18-layer residual network
- **`ResNet34`**: 34-layer residual network  
- **`ResNet50`**: 50-layer residual network with bottleneck blocks
- **`ResNet101`**: 101-layer residual network

#### Embedding Variants
- **`ResNet18Embedding`**: ResNet18 for embedding generation
- **`ResNet34Embedding`**: ResNet34 for embedding generation
- **`ResNet50Embedding`**: ResNet50 for embedding generation
- **`ResNet101Embedding`**: ResNet101 for embedding generation

```python
# Classification model
model = model.ResNet18(numClasses=10, inChannels=1)

# Embedding model
model = model.ResNet18Embedding(embeddingDim=512, inChannels=1)
```

### Data Loaders

#### TripletLoader
Custom data loader for triplet learning with audio data:

```python
# For triplet learning
dataset = dl.TripletLoader(
    rootDir="./data/SpeechCommands/speech_commands_v0.02",
    triplet=True,
    targetLength=16000
)

# For single samples
dataset = dl.TripletLoader(
    rootDir="./data/SpeechCommands/speech_commands_v0.02",
    triplet=False,
    targetLength=16000
)
```

### Loss Functions

```python
# Triplet loss for metric learning
criterion = lossFunction.TripletLoss(margin=0.2)

# Clustering loss with learnable centroids
criterion = lossFunction.TrainableClusteringLoss(
    numClusters=10, 
    embeddingDim=512
)
```

## 💡 Examples

### Complete Training Pipeline (Supervised)

```python
from Supervised import dataloader as dl, model, lossFunction, optimizer, main

# Load data
trainLoader, testLoader, classes = dl.Fashion(batchSize=32)

# Setup model
net = model.fashionModel1()
criterion = lossFunction.CrossEntropyLoss()
optim = optimizer.Adam(net, lr=0.001)

# Train using built-in function
main.trainPrediction(trainLoader, testLoader, optim, net, criterion, epochs=10)
```

### Triplet Learning Pipeline (Unsupervised)

```python
from Unsupervised import dataloader as dl, model, lossFunction, optimizer
import torch

# Setup
dataset = dl.TripletLoader("./data/SpeechCommands/speech_commands_v0.02", triplet=True)
trainLoader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

net = model.ResNet18Embedding(embeddingDim=512)
criterion = lossFunction.TripletLoss(margin=0.2)
optim = optimizer.AdamW(net, lr=0.001)

# Training loop
for epoch in range(5):
    for batch in trainLoader:
        anchor_out = net(batch['anchor'])
        positive_out = net(batch['positive'])
        negative_out = net(batch['negative'])
        
        loss = criterion(anchor_out, positive_out, negative_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
```

## 📚 API Reference

### Supervised Module Structure
```
Supervised/
├── dataloader.py     # Dataset loading utilities
├── model.py          # Neural network architectures
├── lossFunction.py   # Loss function implementations
├── optimizer.py      # Optimizer wrappers
└── main.py           # Training utilities
```

### Unsupervised Module Structure
```
Unsupervised/
├── dataloader.py     # Custom triplet data loaders
├── model.py          # ResNet architectures
├── lossFunction.py   # Triplet and clustering losses
├── optimizer.py      # Optimizer wrappers
└── main.py           # Training examples
```

### Key Classes and Functions

#### Data Loaders
- `Fashion()`: FashionMNIST dataset loader
- `Speech()`: Speech Commands dataset loader  
- `TripletLoader`: Custom triplet sampling for audio data

#### Models
- `fashionModel1/2`: CNN models for FashionMNIST
- `fashionEmbedded`: Embedding model for FashionMNIST
- `speechEmbedded`: Embedding model for speech
- `ResNet18/34/50/101`: ResNet variants for 1D data

#### Loss Functions
- `CrossEntropyLoss()`: Classification loss
- `TripletLoss()`: Metric learning loss
- `TrainableClusteringLoss()`: Unsupervised clustering loss


## 🙏 Acknowledgments

- Special thanks to Mr. Vishal Kumar, CrossCap Labs, IIITD
- Built with PyTorch
- Inspired by various deep learning research papers
- Special thanks to the open-source community

---

For more examples and detailed usage, check out the [Jupyter notebook](TrainingParadigms.ipynb) included in this repository.
