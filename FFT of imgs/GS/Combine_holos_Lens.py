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

height, width=1080,1920
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
arr_r_mod=np.mod(arr_r,2)
arr_g_mod=np.mod(arr_g,2)
arr_b_mod=np.mod(arr_b,2)    

"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*(255/1.85)
arr_g_modified=arr_g_mod*(255/2.63)
arr_b_modified=arr_b_mod*(255/3.55)
"""Convert array to image"""
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
rgb_image[:, :, 0] = arr_r_modified  # Set red channel
rgb_image[:, :, 1] = arr_g_modified
rgb_image[:, :, 2] = arr_b_modified
arr_r_im=Image.fromarray(rgb_image)
# arr_r_im.save(f"Lens_rgb{f}.png")

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/FFT of imgs/GS/lotus_FH_0_1_C_nL.png")
im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/FFT of imgs/GS/lotus_V_1_C_nL.png")

im_rh_array = np.array(im_rh)*255
im_gh_array = np.array(im_gh)*255

# im_new_array = np.zeros_like(im_rh_array)
# im_new_array[:,:,0] = im_rh_array[:,:,0]+im_gh_array[:,:,0]+arr_r_modified
# im_new_array[:,:,1] = im_rh_array[:,:,1]+im_gh_array[:,:,1]+arr_g_modified
# im_new_array[:,:,2] = im_rh_array[:,:,2]+im_gh_array[:,:,2]+arr_b_modified

im_new_array = np.zeros_like(im_rh_array)
im_new_array[:,:,0] = im_rh_array[:,:,0]+im_gh_array[:,:,0]#+arr_r_modified
# im_new_array[:,:,1] = im_rh_array[:,:,1]+im_gh_array[:,:,1]+arr_g_modified
# im_new_array[:,:,2] = im_rh_array[:,:,2]+arr_b_modified+im_gh_array[:,:,2]

im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('Cb_lotus_c_w_H0V0_.png')
# im_new.show()
# # Display results
# plt.figure()
# plt.imshow(I_lens,cmap="hot")
# plt.colorbar()#label='Phase (radians)'
# plt.axis('off')
# plt.title("FFT of Lens for red laser")
# plt.show()

# # # Display results
# plt.figure()
# plt.imshow(arr_g_mod,cmap="Greens")
# plt.colorbar()#label='Phase (radians)'
# plt.axis('off')
# plt.title("Lens for red laser im")
# plt.show()
