# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os

height, width=1920,1920
arr_r=np.zeros((height, width))
arr_g=np.zeros((height, width))
arr_b=np.zeros((height, width))
pixel_size=4.5e-6#4.5e-6
f=2 # meters
# Define wavelengths (in meters, for example)
lambda_r = 0.633e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.450e-6

center_h=height//2
center_w=width//2
"""RGB lens"""
for i in range(height):
    for j in range(width):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        arr_r[i, j] =  -r**2 / (f * lambda_r) #np.pi *
        arr_g[i, j] =  -r**2 / (f * lambda_g)
        arr_b[i, j] =  -r**2 / (f * lambda_b)
"""mod into 0-2"""
arr_r_mod_o=np.mod(arr_r,2)
arr_r_mod=np.where(arr_r_mod_o<1,0,1)
arr_g_mod_o=np.mod(arr_g,2)
arr_g_mod=np.where(arr_g_mod_o<1,0,1)
arr_b_mod_o=np.mod(arr_b,2)
arr_b_mod=np.where(arr_b_mod_o<1,0,1)    

"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*92#(255/1.85)
arr_g_modified=arr_g_mod*145#(255/2.63)
arr_b_modified=arr_b_mod*112#(255/3.55)
# """Convert array to image"""

rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
rgb_image[:, :, 0] = arr_r_modified  # Set red channel
rgb_image[:, :, 1] = arr_g_modified
rgb_image[:, :, 2] = arr_b_modified
# arr_r_im=Image.fromarray(rgb_image)
def crop(im_modify,name):
    y_offset=center_h-1080//2
    im_cropped=im_modify[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"Bi_Lens_rgb{f}.png")
    
C=crop(rgb_image, "c")
# # Display results
plt.figure()
plt.imshow(arr_r_modified,cmap="Reds")
plt.colorbar()#label='Phase (radians)'
plt.axis('off')
plt.title("FFT of Lens for red laser")
plt.show()
plt.figure()
plt.imshow(arr_r_mod,cmap="Reds")
plt.colorbar()#label='Phase (radians)'
plt.axis('off')
plt.title("FFT of Lens for red laser")
plt.show()

# # Display results
plt.figure()
plt.imshow(arr_g_mod,cmap="Greens")
plt.colorbar()#label='Phase (radians)'
plt.axis('off')
plt.title("Lens for green laser im")
plt.show()
plt.figure()
plt.imshow(arr_b_mod,cmap="Blues")
plt.colorbar()#label='Phase (radians)'
plt.axis('off')
plt.title("Lens for blue laser im")
plt.show()