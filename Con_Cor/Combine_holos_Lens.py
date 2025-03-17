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
rdp,gdp,bdp=1.85,2.63,3.55
"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*(255/rdp)
arr_g_modified=arr_g_mod*(255/gdp)
arr_b_modified=arr_b_mod*(255/bdp)
"""Convert array to image"""
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
rgb_image[:, :, 0] = arr_r_modified  # Set red channel
rgb_image[:, :, 1] = arr_g_modified
rgb_image[:, :, 2] = arr_b_modified
arr_r_im=Image.fromarray(rgb_image)
# arr_r_im.save(f"Lens_rgb{f}.png")
t="pi"
im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_V_Bi(pi)_nL n_1.png")
im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_H0_bi(pi)_nL n_1.png")
im_ghex=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_H180_bi(pi) nL n_1.png")
# plt.figure()
# plt.imshow(im_rh)
# plt.show()
# plt.figure()
# plt.imshow(im_gh)
# plt.show()
# plt.figure()
# plt.imshow(im_ghex)
# plt.show()
im_rh_array = np.array(im_rh)*255
im_gh_array = np.array(im_gh)*255

im_new_array = np.zeros_like(im_rh_array)
im_new_array[:,:,0] = im_rh_array[:,:,0]+im_gh_array[:,:,0]+arr_r_modified
im_new_array[:,:,1] = im_rh_array[:,:,1]+im_gh_array[:,:,1]+arr_g_modified
im_new_array[:,:,2] = im_rh_array[:,:,2]+im_gh_array[:,:,2]+arr_b_modified

im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
# im_new.save(f'F_V_H0_bi {t} c.png')

# # im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_V_nL_(pi) n_1.png")
# im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_H0 r nL_(tri) n_1.png")

# # im_rh_array = np.array(im_rh)*255
# im_gh_array = np.array(im_gh)*255

# im_new_array = np.zeros_like(im_rh_array)
# im_new_array[:,:,0] = im_rh_array[:,:,0]+im_gh_array+arr_r_modified
# im_new_array[:,:,1] = im_rh_array[:,:,1]+arr_g_modified
# im_new_array[:,:,2] = im_rh_array[:,:,2]+arr_b_modified

# im_new_array = im_new_array.astype(np.uint8)

# im_new = Image.fromarray(im_new_array)
# im_new.save(f'F_V_H0_{t}_r.png')

# im_new_array_ex = np.zeros_like(im_rh_array)
# im_new_array_ex[:,:,0] = im_gh_array+arr_r_modified
# im_new_array_ex = im_new_array_ex.astype(np.uint8)
# im_new_ex = Image.fromarray(im_new_array_ex)
# im_new_ex.save(f'F_V_H0_{t}_L r.png')

# # im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_V_nL It1_1_It2_0.png")
# im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_H0 g nL_(tri) n_1.png")

# im_rh_array = np.array(im_rh)*255
# im_gh_array = np.array(im_gh)*255

# im_new_array = np.zeros_like(im_rh_array)
# im_new_array[:,:,0] = im_rh_array[:,:,0]+arr_r_modified
# im_new_array[:,:,1] = im_rh_array[:,:,1]+im_gh_array+arr_g_modified
# im_new_array[:,:,2] = im_rh_array[:,:,2]+arr_b_modified

# im_new_array = im_new_array.astype(np.uint8)

# im_new = Image.fromarray(im_new_array)
# im_new.save(f'F_V_H0_{t}_g.png')

# im_new_array_ex = np.zeros_like(im_rh_array)
# im_new_array_ex[:,:,1] = im_gh_array+arr_g_modified
# im_new_array_ex = im_new_array_ex.astype(np.uint8)
# im_new_ex = Image.fromarray(im_new_array_ex)
# im_new_ex.save(f'F_V_H0_{t}_L g.png')

# # im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_V_nL It1_1_It2_0.png")
# im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/F_H0 b nL_(tri) n_1.png")

# # im_rh_array = np.array(im_rh)*255
# im_gh_array = np.array(im_gh)*255

# im_new_array = np.zeros_like(im_rh_array)
# im_new_array[:,:,0] = im_rh_array[:,:,0]+arr_r_modified
# im_new_array[:,:,1] = im_rh_array[:,:,1]+arr_g_modified
# im_new_array[:,:,2] = im_rh_array[:,:,2]+im_gh_array+arr_b_modified

# im_new_array = im_new_array.astype(np.uint8)

# im_new = Image.fromarray(im_new_array)
# im_new.save(f'F_V_H0_{t}_b.png')

# im_new_array_ex = np.zeros_like(im_rh_array)
# im_new_array_ex[:,:,2] = im_gh_array+arr_b_modified
# im_new_array_ex = im_new_array_ex.astype(np.uint8)
# im_new_ex = Image.fromarray(im_new_array_ex)
# im_new_ex.save(f'F_V_H0_{t}_L b.png')
