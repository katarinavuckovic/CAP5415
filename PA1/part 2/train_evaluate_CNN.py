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
        # ----------------- YOUR CODE HERE ----------------------
        #
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
            
            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Compute loss based on same criterion as training 
            criterion = nn.CrossEntropyLoss()  #this criterion combines LogSoftmax and NLLoss in one single class.
            loss = criterion(output,target)
            #F.cross_entropy(output, target)
            #loss = F.nll_loss(output, target, reduction='sum')
            #loss += F.nll_loss(output, target, reduction='sum').item() 
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            '''
            if batch_idx ==100:
              print(pred)
              print(target)
            '''

            # ======================================================================
            # Count correct predictions overall 
            # ----------------- YOUR CODE HERE ----------------------
            #
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
    # ----------------- YOUR CODE HERE ----------------------
    #
    # CrossEntropyLoss combines logSoftMax and nlllos function
    criterion = nn.CrossEntropyLoss() 
    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Use stochastic gradient descent.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate) #, momentum=0.9)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)

    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)

    
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
        # printing time to keep track of the training duration
        elapsed = time.time() - t
        print("Time elapsed:")
        print(elapsed)
        
    print(accuracy_train)
    print(accuracy_test)
    #plotting training loss and accuracy
    fig = plt.figure(1)
    epochs = range(1,FLAGS.num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_test, 'b', label='Validation loss')
    plt.title('Model '+str(FLAGS.mode)+' Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()  
    fig.savefig('LossGraph'+str(FLAGS.mode)+'.jpg')
    fig = plt.figure(2)
    epochs = range(1,FLAGS.num_epochs + 1)
    plt.plot(epochs, accuracy_train*100, 'g', label='Training accuracy')
    plt.plot(epochs, accuracy_test, 'b', label='Validation accuracy')
    plt.title('Model '+str(FLAGS.mode)+' Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy in %')
    plt.legend()
    plt.show()  
    fig.savefig('AccuracyGraph_mode'+str(FLAGS.mode)+'.jpg')
  
    # Displaying accuracy and  
    print("accuracy is {:2.2f}".format(best_accuracy))
    print("converge epoch is {:2.2f}".format(best_epoch))
    print("Training and evaluation finished")
    elapsed = time.time() - t
    print("Time elapsed:")
    print(elapsed)
   
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1, #0.1 for model 1-2, 0.03 for model 3-5
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60, # 60 for model 1-4, 40 for model 5
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    # Iterate over all five modes
    for mode in range(1,6):
      print(20 * '*' + 'Model '+str(mode)+20 * '*' )
      if mode==1:
        FLAGS.mode = mode
      if mode==2:
        FLAGS.mode = mode
      if mode==3:
        FLAGS.mode = mode
        FLAGS.learning_rate = 0.03
      if mode==4:
        FLAGS.mode = mode
        FLAGS.learning_rate = 0.03
      if mode==5:
        FLAGS.mode = mode
        FLAGS.learning_rate = 0.03
        FLAGS.num_epochs = 40
      run_main(FLAGS)


    
    
        

      
    
    