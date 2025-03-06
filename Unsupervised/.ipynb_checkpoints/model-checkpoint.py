import torch
from torch.autograd.function import _SingleLevelFunction
import torch.nn as nn
import torch.nn.functional as F

# Basic Block for ResNet18,34
class BasicBlock(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, inChannels, outChannels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outChannels)

        self.conv2 = nn.Conv1d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outChannels)

        self.downsample = downsample  # This helps adjust dimensions when needed

    def forward(self, x):
        residual = x  # Store input for residual connection

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add residual connection
        out = F.relu(out)

        return out

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self,inChannels,outChannels,stride = 1,downsample = None):
        super(BottleneckBlock,self).__init__()

        self.conv1 = nn.Conv1d(inChannels,outChannels,kernel_size = 1,stride = 1)
        self.bn1 = nn.BatchNorm1d(outChannels)

        self.conv2 = nn.Conv1d(outChannels,outChannels,kernel_size = 3,stride = stride,padding = 1)
        self.bn2 = nn.BatchNorm1d(outChannels)

        self.conv3 = nn.Conv1d(outChannels,outChannels*self.expansion,kernel_size = 1,stride = 1)
        self.bn3 = nn.BatchNorm1d(outChannels*self.expansion)

        self.downsample = downsample

    def forward(self,x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = residual + out

        return out
        
class ResNet(nn.Module):
    def __init__(self,block,layers,inChannels = 1,numClasses = 1000):
        super(ResNet,self).__init__()

        #initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(inChannels,64,kernel_size = 7,stride = 2,padding = 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.inChannels = 64
        self.layer1 = self.makeLayer(block,64,layers[0],stride=1) # outChannels = 64
        self.layer2 = self.makeLayer(block,128,layers[1],stride=2) # outChannels = 128
        self.layer3 = self.makeLayer(block,256,layers[2],stride=2) # outChannels = 256
        self.layer4 = self.makeLayer(block,512,layers[3],stride=2) # outChannels = 512

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*block.expansion,numClasses)

        self.initialiseWeights()

    def makeLayer(self, block, outChannels, numBlocks, stride=1):
        downsample = None
        if stride != 1 or self.inChannels != outChannels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inChannels, outChannels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outChannels * block.expansion)
            )
    
        layers = []
        layers.append(block(self.inChannels, outChannels, stride, downsample))
        self.inChannels = outChannels * block.expansion  # Fix: Update inChannels correctly
    
        for _ in range(1, numBlocks):
            layers.append(block(self.inChannels, outChannels))  # Fix: Correct inChannels usage
    
        return nn.Sequential(*layers)


    def initialiseWeights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x

def ResNet18(numClasses,inChannels):
    return ResNet(BasicBlock,[2,2,2,2],inChannels,numClasses)

def ResNet50(numClasses,inChannels):
    return ResNet(BottleneckBlock,[3,4,6,3],inChannels,numClasses)

def ResNet101(numClasses,inChannels):
    return ResNet(BottleneckBlock,[3,4,23,3],inChannels,numClasses)



        