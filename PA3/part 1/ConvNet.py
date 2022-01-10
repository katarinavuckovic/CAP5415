import time
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Mode 1: 3 conv layers & 3 FC layers
         #------------------------------------------------
        # ConV Layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

         # Mode 2: 4 conv layers & 3 FC layers
         #------------------------------------------------
        # ConV Layers
        self.conv_1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv_2 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv_3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv_4 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc_1 = nn.Linear(64* 2*2, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 10)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        

    def model_1(self, x):
        # ======================================================================
        # 3 conv layers & 3 FC layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x     

    
    def model_2(self, x):
        # ======================================================================
         # 4 conv layers & 3 FC layers
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        x = self.pool(F.relu(self.conv_4(x)))
        # flattening
        x = x.view(-1, 64 * 2 * 2)
        # fully connected layers
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x
  