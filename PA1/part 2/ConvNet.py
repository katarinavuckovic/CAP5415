import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        # Define network parameters 
        hidden_layers1 = 100 
        hidden_layers2 = 1000
        kernels = 40
        kernel_size = 5
        num_classes =10
        
        # Define various layers here, such as in the tutorial example
        # ConV Layers
        self.conv1 = nn.Conv2d(1,kernels,kernel_size,stride =1)
        self.conv2 = nn.Conv2d(kernels,kernels, kernel_size,stride =1)
        # MaxPool Layers
        self.maxpool1 = nn.MaxPool2d((2,2),stride= (2,2))
        self.maxpool2 = nn.MaxPool2d((2,2),stride= (2,2))
        # Hidden FC Layers
        self.fc1 = nn.Linear(784,hidden_layers1) #model 1
        self.fc2 = nn.Linear(640, hidden_layers1) #model 2-4
        self.fc3 =  nn.Linear(hidden_layers1, hidden_layers1)  #model 4
        self.fc4 =  nn.Linear(640, hidden_layers2) #model 5
        self.fc5 =  nn.Linear(hidden_layers2, hidden_layers2) #model 5
        # Last FC layer 
        self.fc_1 = nn.Linear(hidden_layers1, num_classes) #model 1-4
        self.fc_2 = nn.Linear(hidden_layers2, num_classes) #model 5
        #self.fc4 =  nn.Linear(640, hidden_layers)
        self.dropout1 = nn.Dropout(0.5)
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer, with sigmoid activation function. 
        # Returns fully connected layer
        #
        # ----------------- YOUR CODE HERE ----------------------
        
        X = torch.reshape(X, [-1, 784])
        X = self.fc1(X)
        X = F.sigmoid(X)
        X = self.fc_1(X)
        return X     

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer, with sigmoid.
        #
        # ----------------- YOUR CODE HERE ----------------------
        # 1st Conv Layer
        X = self.conv1(X)
        X = torch.sigmoid(X)
        X = self.maxpool1(X)
        # 2nd Conv Layer
        X = self.conv2(X)
        X = torch.sigmoid(X)
        X = self.maxpool2(X)
       # print(X.shape)
        #X = X.view(-1,640)
        #print('after view:')
        # FC Layer        
        X = torch.reshape(X, [-1,640])
        X = self.fc2(X)
        X = F.sigmoid(X)
        X = self.fc_1(X)
       # X = F.log_softmax(X, dim = 1)
        return X
    # Replace sigmoid with ReLU.(X, dim)
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #Note: need to change LR to 0.03 for this part
        # 1st Conv Layer
        X = self.conv1(X)
        X = F.relu(X)
        X = self.maxpool1(X)
        # 2nd Conv Layer
        X = self.conv2(X)
        X = F.relu(X)
        X = self.maxpool2(X)
        # FC Layer
        X = torch.reshape(X, [-1, 640])
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc_1(X)
       # X = F.log_softmax(X, dim = 1)
        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        # 1st Conv Layer
        X = self.conv1(X)
        X = F.relu(X)
        X = self.maxpool1(X)
        # 2nd Conv Layer
        X = self.conv2(X)
        X = F.relu(X)
        X = self.maxpool2(X)
        # 1st FC Layer
        X = torch.reshape(X, [-1, 640])
        X = self.fc2(X)       
        X = F.relu(X)
        # 2nd FC Layer
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc_1(X)
        return X
        

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        # 1st Conv Layer
        X = self.conv1(X)
        X = F.relu(X)
        X = self.maxpool1(X)
        # 2nd Conv Layer
        X = self.conv2(X)
        X = F.relu(X)
        X = self.maxpool2(X)
        # 1st FC Layer
        X = torch.reshape(X, [-1, 640])
        X = self.fc4(X)       
        X = F.relu(X)
        X = self.dropout1(X)
        # 2nd FC Layer
        X = self.fc5(X)
        X = F.relu(X)
        X = self.dropout1(X)
        X = self.fc_2(X)
        return X
    
    
