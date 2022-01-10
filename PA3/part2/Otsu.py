'''
Description:
This program implements image segmentation based on Otsu Tresholding 


Author: Katarina Vuckovic, University of Central Florida
Date:11/28/2021
'''

from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
import scipy

#Otsu Tresholding
# Load image
I = plt.imread('gray7.jpg')
gray = I
pixel_number = gray.shape[0] * gray.shape[1]
mean_weigth = 1.0/pixel_number
# Compute histogram
his, bins = np.histogram(gray, np.array(range(0, 256)))
#set initial values
final_thresh = -1
final_value = -1
for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
    # Compute probabilities
    Wb = np.sum(his[:t]) * mean_weigth
    Wf = np.sum(his[t:]) * mean_weigth
    # Compute means
    mub = np.mean(his[:t])
    muf = np.mean(his[t:])
    # In-between class variance
    value = Wb * Wf * (mub - muf) ** 2
    # Print intermediate steps for analysis
  # print("Wb", Wb, "Wf", Wf)
  # print("t", t, "value", value)

    # Check if new variance is greater than the previos max variance 
    # and if it is chagne new max variance
    if value > final_value:
        final_thresh = t
        final_value = value
        s
final_img = gray.copy()
# Print Otsu threshold
print(final_thresh)
final_img[gray > final_thresh] = 255
final_img[gray < final_thresh] = 0
# Display image
imgplotgrey = plt.imshow(final_img,cmap='gray') #gray scalepl