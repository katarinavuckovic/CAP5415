# Katarina Vuckovic
# Oct 2021
# CAP5415 Computer Vision
# Programming Assignment 1 Part 1

from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
import scipy
from scipy import ndimage
from skimage import data, filters

def my_convolve2d(image, kernel):
    """
    This function which takes an image and a kernel and returns the convolution of them.

    :param image: a numpy array of size [image_height, image_width].
    :param kernel: a numpy array of size [kernel_height, kernel_width].
    :return: a numpy array of size [image_height, image_width] (convolution output).
    """
    # convolution output
    size = int(np.max(kernel.shape)/2)
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2*size, image.shape[1] + 2*size))
    padsize =2*(size)-1
    image_padded[1:-padsize, 1:-padsize] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
           # print(kernel.shape)
            output[y, x]=(kernel * image_padded[y: (y+kernel.shape[0]), x:(x+kernel.shape[1]) ]).sum()

    return output

    
def gaussian(size, sigma):
    '''
     1D gaussian mask
     inputs: 
     size - size of filter
     sigma - STD (sigma) of gaussian 
     output: 1D gaussian of size "size"
     '''
    size = int(size) // 2
    x = np.arange(-size,size+1,1)
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-(x*x)/(2*sigma*sigma))*normal
    return g

def dgaussian(size, sigma):
    '''
     1D gaussian derivate mask
     inputs: 
     size - size of filter
     sigma - STD (sigma) of gaussian 
     output: 1D gaussian derivative of size "size"
     '''
    size = int(size) // 2
    x = np.arange(-size,size+1,1)
    normal = -x / (2.0 * np.pi * sigma*sigma*sigma)
    g = np.exp(-(x*x)/(2*sigma*sigma))*normal
    return g

def magnitude(Ix_prime,Iy_prime):
    '''
    Calculate magnitude by combining x and y components
    '''
    M = np.sqrt(np.multiply(Ix_prime,Ix_prime)+np.multiply(Iy_prime,Iy_prime))
    #M = np.sqrt(Ix_prime**2+Iy_prime**2)
    M = np.round(M/np.max(M)*255) 
    return M

def normalize(I):
    '''
    Due to the difference on value scale,
    We need Normalization on Image scale [0..255]
    '''
    low = I.min()
    high = I.max()
    I = (I-low) * 1.0 / (high - low)
    return I * 255

#Step 7: NMS
def non_maximal_suppressor(grad_mag, closest_dir) :
    '''
    Implementation of NMS on image
    Inputs: 
    grad_mag - magnitude of image
    cosest_dir - estimated direction from atan(Y/x)
    Output:
    thinned out image 
    '''
    thinned_output = np.zeros(grad_mag.shape)
    for i in range(1, int(grad_mag.shape[0] - 1)) :
        for j in range(1, int(grad_mag.shape[1] - 1)) :
            
            if(closest_dir[i, j] == 0) :
                if((grad_mag[i, j] > grad_mag[i, j+1]) and (grad_mag[i, j] > grad_mag[i, j-1])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0
            
            elif(closest_dir[i, j] == 45) :
                if((grad_mag[i, j] > grad_mag[i+1, j+1]) and (grad_mag[i, j] > grad_mag[i-1, j-1])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0
            
            elif(closest_dir[i, j] == 90) :
                if((grad_mag[i, j] > grad_mag[i+1, j]) and (grad_mag[i, j] > grad_mag[i-1, j])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0
            
            else :
                if((grad_mag[i, j] > grad_mag[i+1, j-1]) and (grad_mag[i, j] > grad_mag[i-1, j+1])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0
            
    return thinned_output

#step 8: Hystersis Thresholding
def hysteresis(img, weak, strong):
    '''
    Implementation of hystersis thresholding on image
    Inputs: 
    img - image that you want to apply the hystersis thresholding on
    weak,strong - low and high tresholds for the hystersis
    Output:
    img - image after applying hystersis thresholding
    '''
    weak = int(weak*255)
    strong = int(strong*255)
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] > strong):
              img[i,j]=255
            elif (img[i,j] < weak):
              img[i,j]=0
            else:
               if ((img[i+1,j]>strong) or (img[i-1,j]>strong) or (img[i,j-1]>strong) 
                  or (img[i,j+1]>strong) or (img[i+1,j-1]>strong) 
                  or (img[i-1,j-1]>strong) or (img[i+1,j+1]>strong) or (img[i-1,j+1]>strong)):
                 img[i,j]=255
               else:
                img[i,j]=0   
            
    return img

im_name = 'gray1' #name of input .jpg image
s = [1, 5, 10] # set to the desired sigma values
size= 4# select the size of the filter
# Create a loop to evaluate the output for the different sigma values
for sigma in s:

    # Step 1: read image, store image and display grayscale
    I=plt.imread(im_name+'.jpg')

    # Step 2: create 1D Gaussian mask (G)
    g = gaussian(size, sigma)
    g = g.reshape(1,-1)

    # Step 3: create derivate of gaussian masks Gx and Gy 
    gx = dgaussian(size, sigma)
    gx = gx.reshape(1,-1)

    # Step 4: convolve Gaussian (G) with I along x & y to get Ix & Iy 
    Iy = my_convolve2d(I, np.transpose(g))
    Ix = my_convolve2d(I, g)  
    # Normalize output
    Iy = normalize(Iy)
    Ix = normalize(Ix)

    # Step 5: obtain Ix' and Iy' the derivative of filted image
    Iy_prime = my_convolve2d(Iy, np.transpose(gx))
    Ix_prime = my_convolve2d(Ix, gx)
   # Iy_prime = normalize(Iy_prime)
   # Ix_prime = normalize(Ix_prime)

    # Step 6: compute the magnitude (M) and angle
    M = magnitude(Ix_prime,Ix_prime)
    M = normalize(M)
    #M = M.astype(int)
    angle = np.degrees(np.arctan2(Iy_prime, Ix_prime))

    # Step 7: non-maximum supression
    Inms = non_maximal_suppressor(M, angle)
    I = np.round(I/np.max(I)*255) 

    # Step 8: Hystersis Thresholding
    # low and high are weak and strong limits
    low = 0.1 
    high = 0.3
    I_hyst = hysteresis(Inms, low, high)

    # Plot and save figures
    iplt = plt.imshow(I,cmap='gray')
    plt.axis('off') 
    plt.savefig('I_result.jpg')
    ixplot = plt.imshow(Ix,cmap='gray') 
    plt.axis('off') 
    plt.savefig('Ix_result_'+im_name+'_'+str(sigma)+'.jpg')
    iyplot = plt.imshow(Iy,cmap='gray') 
    plt.axis('off') 
    plt.savefig('Iy_result_'+im_name+'_'+str(sigma)+'.jpg')
    ixxplot = plt.imshow(Ix_prime,cmap='gray') 
    plt.axis('off') 
    plt.savefig('Ixx_result_'+im_name+'_'+str(sigma)+'.jpg')
    iyyplot = plt.imshow(Iy_prime,cmap='gray') 
    plt.axis('off') 
    plt.savefig('Iyy_result_'+im_name+'_'+str(sigma)+'.jpg')
    implot = plt.imshow(M,cmap='gray') 
    plt.axis('off') 
    plt.savefig('M_result_'+im_name+'_'+str(sigma)+'.jpg')
    inmsplot = plt.imshow(Inms,cmap='gray') 
    plt.axis('off') 
    plt.savefig('Inms_result_'+im_name+'_'+str(sigma)+'.jpg')
    ihysplot = plt.imshow(I_hyst,cmap='gray') 
    plt.axis('off') 
    plt.savefig('Ihyst_result_'+im_name+'_'+str(sigma)+'.jpg')

