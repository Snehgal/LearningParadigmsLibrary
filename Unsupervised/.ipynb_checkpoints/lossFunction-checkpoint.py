import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Clustering loss
class TrainableClusteringLoss(nn.Module):
    """
    A trainable clustering loss function that optimizes cluster centroids.
    
    Attributes:
        numClusters (int): Number of clusters.
        embeddingDim (int): Dimensionality of the embedding space.
        centroids (nn.Parameter): Learnable cluster centroids.
    """
    def __init__(self, numClusters=10, embeddingDim=512):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(numClusters, embeddingDim))  # Initialize random cluster centers
        
    def forward(self, embeddings):
        """
        Computes the clustering loss by assigning embeddings to the nearest centroid.
        
        Args:
            embeddings (Tensor): Input embeddings.
        
        Returns:
            tuple: Mean squared error loss and assigned cluster indices.
        """
        # Compute L2 distance from all cluster centers
        distances = torch.cdist(embeddings, self.centroids, p=2)

        # Find the closest cluster center for each embedding
        clusterAssigned = torch.argmin(distances, dim=1)  # Index of the closest centroid

        # Calculate MSE loss between embeddings and their assigned cluster centers
        loss = F.mse_loss(embeddings, self.centroids[clusterAssigned])
        return loss, clusterAssigned
    
# Triplet loss for metric learning
class TripletLoss(nn.Module):
    """
    Triplet loss function to encourage embeddings of similar samples to be closer together
    while pushing apart embeddings of dissimilar samples.
    
    Attributes:
        margin (float): Margin for the triplet loss function.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Computes the triplet loss.
        
        Args:
            anchor (Tensor): Anchor embedding.
            positive (Tensor): Positive sample embedding.
            negative (Tensor): Negative sample embedding.
        
        Returns:
            Tensor: The computed triplet loss.
        """
        # Compute L2 distance between anchor-positive and anchor-negative pairs
        posDist = F.pairwise_distance(anchor, positive, p=2)
        negDist = F.pairwise_distance(anchor, negative, p=2)

        # Compute triplet loss with margin
        loss = torch.clamp(posDist - negDist + self.margin, min=0.0).mean()
        return loss