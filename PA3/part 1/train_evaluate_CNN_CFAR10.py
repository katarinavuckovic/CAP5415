'''
Description:
This code trains and tests a CNN classification netowrk for the CFAR-10 Dataset.

Author: Katrina Vuckovic, University of Central Florida
Date: 11/28/2021 
'''

from __future__ import print_function
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import time

from ConvNet import ConvNet 

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        #target = torch.argmax(target, dim=1)
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(output,target)
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        # ----------------- YOUR CODE HERE ----------------------
        #
        correct += pred.eq(target.view_as(pred)).sum().item()
    # Display training results for each epoch    
    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    return train_loss, train_acc
    


def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
          
            # Compute loss based on same criterion as training 
            criterion = nn.CrossEntropyLoss()  #this criterion combines LogSoftmax and NLLoss in one single class.
            loss = criterion(output,target)

            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy


def run_main(FLAGS):
    t = time.time()
    print('Mode')
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    
    criterion = nn.CrossEntropyLoss() 
    # ======================================================================
    # Define optimizer function.
   
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate) #, momentum=0.9)
        
    
    # Create transformations to apply to each data sample 
   
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Import CFAR10 Dataset
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size= FLAGS.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size= FLAGS.batch_size,shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

      
    # Inititalizing parameters
    best_epoch = 0.0
    best_accuracy = 0.0
    loss_train = np.zeros(FLAGS.num_epochs)
    loss_test = np.zeros(FLAGS.num_epochs)
    accuracy_train = np.zeros(FLAGS.num_epochs)
    accuracy_test = np.zeros(FLAGS.num_epochs)
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(20 * '*', 'epoch', epoch, 20 * '*')
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
                                            
        #train_loss, train_accuracy = train(model, device, train_loader,
        #                                    optimizer, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader)

        loss_train[epoch-1]= train_loss
        accuracy_train[epoch-1]= train_accuracy
        loss_test[epoch-1]=test_loss
        accuracy_test[epoch-1]= test_accuracy
        
        #saving best accuracy and the epoch at which it occured
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch

    
    print('train accuracy:')   
    print(accuracy_train)
    print('test accuracy:') 
    print(accuracy_test)

    # ----------------------------------------------------------------------------
    # total accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    y_pred = []
    y_true = []
    # iterate over test data
    for inputs, labels in test_loader:
            output =model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # Classes
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    '''
    # Build confusion matrix
    plt.figure(1)
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix/np.sum(cf_matrix) *10)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    #plt.xlabel("prediction")
    #plt.ylabel("label (ground truth)")
    #plt.show()
    plt.savefig('conf_mat'+ str(FLAGS.mode)+'.jpg')
    '''
    #plotting training loss and accuracy
    fig1 = plt.figure(2)
    epochs = range(1,FLAGS.num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_test, 'b', label='Validation loss')
    plt.title('Model '+str(FLAGS.mode)+' Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()  
    fig1.savefig('loss_CFAR10_model'+str(FLAGS.mode)+'.jpg')

    fig2 = plt.figure(3)
    #epochs = range(1,FLAGS.num_epochs + 1)
    plt.plot(epochs, accuracy_train*100, 'g', label='Training accuracy')
    plt.plot(epochs, accuracy_test, 'b', label='Validation accuracy')
    plt.title('Model '+str(FLAGS.mode)+' Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy in %')
    plt.legend()
    plt.show()  
    fig2.savefig('accuray_CFAR10_model'+str(FLAGS.mode)+'.jpg') 

  
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01, #0.1 for model 1-2, 0.03 for model 3-5
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',default = 30,
                        type=int,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=20,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
   
    # Iterate over two modes 
    for mode in range(1,3):
      print(20 * '*' + 'Model '+str(mode)+20 * '*' )
      if mode==1:
        FLAGS.mode = mode
      if mode==2:
        FLAGS.mode = mode
      run_main(FLAGS)

    
     


    
    
        

      
    
    