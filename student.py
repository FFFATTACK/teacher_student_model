import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import math

#5-layer CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        hidden_dim = 64 * 4 * 4
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
def simplecnn():
    return SimpleCNN()

#4 layer CNN narrower
class SimpleCNN1(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN1, self).__init__()
        hidden_dim = 64 * 4 * 4
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, dilation=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
def simplecnn1():
    return SimpleCNN1()

##7-layer CNN
class SimpleCNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN2, self).__init__()
        hidden_dim = 64 * 4 * 4
        self.feature = nn.Sequential(
            # 3*32*32
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(4, 4), stride=2, padding=1), # (4*4*3+1)*8=392
            nn.ReLU(),
            # 8*16*16
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 4), stride=2, padding=1), # (4*4*8+1)*8=1032
            nn.ReLU(),
            # 8*8*8
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # (4*4*8+1)*16=2064
            nn.ReLU(),
            # 16*4*4
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1), # (3*3*16+1)*32=4640
            nn.ReLU(),
            # 32*4*4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1), # (3*3*32+1)*64=18496
            nn.ReLU(),
            # 64*4*4
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
def simplecnn2():
    return SimpleCNN2()

#4-layer cnn wider
class SimpleCNN3(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN3, self).__init__()
        hidden_dim = 128 * 8 * 8
        self.feature = nn.Sequential(
            # 3*32*32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1), # (4*4*3+1)*64=3136
            nn.ReLU(),
            # 64*16*16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1), # (4*4*64+1)*256=262400
            nn.ReLU(),
    
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
def simplecnn3():
    return SimpleCNN3()



# MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=32*32*3, num_classes=10):
        super(SimpleMLP, self).__init__()
        hidden_dim = 4096
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)    

def simplemlp():
    return SimpleCNN()

