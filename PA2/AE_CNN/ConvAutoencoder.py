from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


#Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  
        self.conv2 = nn.Conv2d(4, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(16, 4, 3 , padding=1)  
        self.t_conv2 = nn.ConvTranspose2d(4, 1, 3,  padding=1)  
        self.t_conv3= nn.ConvTranspose2d(1, 1, 3, padding=1) 

        self.upsample2x = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        #Encode 
        x = F.relu(self.conv1(x))
        #print(np.shape(x))
        x = self.pool(x)
        #print(np.shape(x))

        x = F.relu(self.conv2(x))
       # print(np.shape(x))
        x = self.pool(x)
       # print(np.shape(x))
        #Decode
        x = F.relu(self.t_conv1(x))
        #print(np.shape(x))
        x = F.relu(self.upsample2x(x))

       # print(np.shape(x))
        x = F.relu(self.t_conv2(x))
       # print(np.shape(x))
        x = F.relu(self.upsample2x(x))

       # print(np.shape(x))
        x = F.sigmoid(self.t_conv3(x))
      #  print(np.shape(x))
              
        return x