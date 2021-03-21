import argparse
import cv2
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt


#import image
#parser = argparse.ArgumentParser(description='Remove dark layer')
#parser.add_argument('-i','--input',type=str, required=True, help='the input tif 16bit image')
#parser.add_argument('-o','--output', type=str, required=True, help='the output tif 16bit image')
#args = parser.parse_args()

direc = "red.tif"

#%%
img = cv2.imread(direc,-1)

plt.imshow(img)
plt.title("Original image")
plt.show()


def better_histogram(image, nbins=1000):
    """Prepare data for the 16bit"""
    data= np.hstack(image)        
    hist, bin_edges = np.histogram(data, nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.


    return hist, bin_centers

def otsu(hist,bin_centers, nbins=1000):
    "Applu Otsu's algorithm to create mask between dark and bright regions"""
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

#%%
    
#Building the mask    
data = better_histogram(img) 
thresh_otsu = otsu(data[0],data[1])    
mask = np.uint16(img<thresh_otsu)    

#Morphological processing ... opening
strel = morph.disk(20)
mask_dark = cv2.morphologyEx(mask,cv2.MORPH_OPEN,strel)
mask_bright = 1 - mask_dark

#Generating 2 arrays and removing all the zeros from the mask for each region
array_dark = np.hstack(mask_dark*img)
array_bright = np.hstack(mask_bright*img)

array_dark = array_dark[array_dark!=0]
array_bright = array_bright[array_bright!=0]

#Apply constrast streching to match the dark side to the bright side
A = np.min(array_bright)
B = np.max(array_bright)

C = np.min(array_dark)
D = np.max(array_dark)

new_dark = (img-C)*((B-A)/(D-C)) + A + 1000
new_image = new_dark*mask_dark + img*mask_bright


plt.imshow(new_image)
plt.title("Rescaled image")
plt.show()


