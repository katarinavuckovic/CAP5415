'''
Description:
This program implements the binary image segmentation.


Author: Katarina Vuckovic, University of Central Florida
Date:11/28/2021
'''

from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
import scipy

#load test image
I=plt.imread('gray7.jpg')
imgplot = plt.imshow(I) #color plot
imgplotgrey = plt.imshow(I,cmap='gray') #gray scaleplot
print(np.max(I))
print(np.min(I))

#plot histogram to find the treshold
%matplotlib inline
plt.hist(I, density = True, bins=25)  
plt.ylabel('Probability')
plt.xlabel('Data');

#implement binary
T = 150# treshold based on histogram

final_img[I> T] = 255
#This section is only used during testing if you want to observe the Histogram after the hi value pixels have been set to max
'''
plt.hist(final_im, density = True, bins=25)  
plt.ylabel('Probability')
plt.xlabel('Data');
plt.xlim([0, 255])
'''
final_img[I < T] = 0
#This section is only used during testing if you want to observe the Histogram after binary is complete
'''
plt.hist(final_im, density = True, bins=25)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data');
plt.xlim([0, 255]
'''
imgplotgrey = plt.imshow(final_img,cmap='gray') #gray scale