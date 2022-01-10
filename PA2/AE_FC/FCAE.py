# CP5415 Computer Vision
# Programming Assigment 2
# Implementing fully connected autoencoder for MNIST
# Date: 11/9/2021
# Author: Katarina Vuckovic 

from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import statistics
import argparse
import time
import os
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

from FullyConnectedAutoencoder import FullyConnectedAutoencoder

# Initialize 
batch_size = 100
n_iters = 5000
num_epochs = 10

# Load MNIST datasets for training and testing
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data/', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle=False) 

# Initialize model
model = FullyConnectedAutoencoder()
print(model)

# Loss Class MSE 
criterion = nn.MSELoss()
# Using basic Adam optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


# Train Model 
for epoch in range(1, num_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data in train_loader:
        images, _ = data
        images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
          
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
output =  model(images.view(-1,28*28))


samples1 = np.array([3,2,1,18,4,8,11,0,84,7]) 
samples2 = np.array([25,5,35,32,6,15,21,17,61,9])

# Sample set 1
i = 0
for sample_num in samples1:
  i+=1
  raw_img = images[sample_num]
  show_img = raw_img.numpy().reshape(28, 28)
  label = labels[sample_num]
  #print(f'Label {label}')
  plt.subplot(2,10,i)
  plt.imshow(show_img, cmap='gray')
  plt.axis('off')
  output_img =  output[sample_num]
  output_img = torch.reshape(output_img,(28,28))
  output_img = output_img.detach().numpy()
  plt.subplot(2,10,i+10)
  plt.imshow(output_img,cmap='gray')
  plt.axis('off')
plt.savefig('FCAE_results1.png')


# Sample set 2
i = 0
for sample_num in samples2:
  i+=1
  raw_img = images[sample_num]
  show_img = raw_img.numpy().reshape(28, 28)
  label = labels[sample_num]
  #print(f'Label {label}')
  plt.subplot(2,10,i)
  plt.imshow(show_img, cmap='gray')
  plt.axis('off')
  output_img =  output[sample_num]
  output_img = torch.reshape(output_img,(28,28))
  output_img = output_img.detach().numpy()
  plt.subplot(2,10,i+10)
  plt.imshow(output_img,cmap='gray')
  plt.axis('off')
plt.savefig('FCAE_results2.png')