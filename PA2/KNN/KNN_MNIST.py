# CAP5415 Computer Vision
# Programmin Assignment 2, Question 1
# KNN Classifier for MNIST Dataset
# Date: 11/9/2021
# Author: Katarina Vuckovic

from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
import scipy
from scipy import ndimage
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits



# Define L2-norm function that is used to calculate the distance between two samples. 
def l2norm(actual_value,predicted_value):
    l2 = np.sum(np.power((actual_value-predicted_value),2))
    return (l2)


#implementation of NN 
def NN(X_test,X_train,y_train):
    predicted_value = X_test
    i = 0
    l2 = np.zeros([len(X_train),1])
    for train in X_train:
      l2[i] = l2norm(train,predicted_value)
      i +=1
    #print(l2)
    predicted_digit = y_train[np.argmin(l2)]
    return predicted_digit

#implemenration of KNN for K>=1
def KNN(X_test,X_train,y_train,K):
  predicted_value = X_test
  i = 0
  l2 = np.zeros([len(X_train),1])
  predicted_digit = np.zeros([K,1])
  for train in X_train:
    l2[i] = l2norm(train,predicted_value)
    i +=1
  #print(l2)'
  min_index = l2[:,0].argsort()[:K]
  pred_k_digits = y_train[min_index]
  (unique, counts) = np.unique(pred_k_digits, return_counts=True)
  if (np.max(counts)>0):
    pred = unique[np.argmax(counts)]
  else:
    print('error')
  return pred


# Import data and split into testing and training datasets
digits = load_digits()
# flatten the images
n_samples = len(digits.images)
print(n_samples)
data = digits.images.reshape((n_samples, -1))
# Split data such that testing is 500 samples
# 500/1797 = 0.2777 (note: n_samples = 197)
X_train, X_test, y_train, y_test = train_test_split( data, digits.target, test_size=0.2777, shuffle=False)


# Calculating accuracy for NN (K=1)
result = np.zeros([len(y_test),1]) 
pred = np.zeros([len(y_test),1]) 
#print(result)
for i in range(len(y_test)):
    pred[i] = NN(X_test[i,:],X_train,y_train)
    if(pred[i]==y_test[i]):
      result[i] = 1
    else:
      result[i] = 0

percent = np.sum(result)/500
print('percentage correct for NN (k=1):',percent)

# calculate accuracy for k>1
result = np.zeros([len(y_test),1]) 
pred = np.zeros([len(y_test),1]) 
K = 7
for k in range(1,K+1):
  percent = 0
  print(k)
  for i in range(len(y_test)):
      pred[i] = KNN(X_test[i,:],X_train,y_train,k)
      if(pred[i]==y_test[i]):
        result[i] = 1
      else:
        result[i] = 0
  percent = np.sum(result)/500
  print('percentage correct for k =',k,': ',percent)
