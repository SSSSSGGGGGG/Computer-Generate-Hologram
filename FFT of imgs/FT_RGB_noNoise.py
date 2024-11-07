# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image
import numpy as np
import scipy as sp
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import matplotlib.pyplot as plt
import os
from skimage import color
import cv2

os.chdir("C:/Users/Laboratorio/MakeHologram/FFT of imgs")
filename="whiteRing_shift"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]

# Define wavelengths (in meters, for example)
lambda_r = 0.633e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.450e-6  # Blue wavelength

#R
im_shift_r=fftshift(im[:,:,0])
#G
im_shift_g=fftshift(im[:,:,1])
#B
im_shift_b=fftshift(im[:,:,2])

    
#R
im_r_rand=im_shift_r
im_rr_ft=fftshift(fft2(im_r_rand))
phase_rr = np.angle(im_rr_ft)
phase_rr_modi=(np.angle(im_rr_ft)/np.pi+1)*(255/1.85)
phase_rr_modi_mod=np.mod(phase_rr_modi,255)

#G
im_g_rand=im_shift_g
im_gr_ft=fftshift(fft2(im_g_rand))
phase_gr = np.angle(im_gr_ft)
phase_gr_modi=(np.angle(im_gr_ft)/np.pi+1)*(255/2.63)
phase_gr_modi_mod=np.mod(phase_gr_modi,255)

#B
im_b_rand=im_shift_b
im_br_ft=fftshift(fft2(im_b_rand))
phase_br = np.angle(im_br_ft)
phase_br_modi=(np.angle(im_br_ft)/np.pi+1)*(255/3.55)
phase_br_modi_mod=np.mod(phase_br_modi,255)

"""Lens"""
arr_r=np.zeros((height, width))
arr_g=np.zeros((height, width))
arr_b=np.zeros((height, width))
pixel_size=4.5e-6#4.5e-6
f=-2 # meters
center_h=height//2
center_w=width//2
"""RGB lens"""
for i in range(height):
    for j in range(width):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        arr_r[i, j] =  r**2 / (f * lambda_r) #np.pi *
        arr_g[i, j] =  r**2 / (f * lambda_g)
        arr_b[i, j] =  r**2 / (f * lambda_b)
"""mod into 0-2"""
arr_r_mod=np.mod(arr_r,2)
arr_g_mod=np.mod(arr_g,2)
arr_b_mod=np.mod(arr_b,2)    

"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*(255/1.85)
arr_g_modified=arr_g_mod*(255/2.63)
arr_b_modified=arr_b_mod*(255/3.55)

im_modify0 = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify0[:,:,0] = phase_rr_modi_mod
im_modify0[:,:,1] = phase_gr_modi_mod
im_modify0[:,:,2] = phase_br_modi_mod
im_modify0 = im_modify0.astype(np.uint8)
im_modi0 = Image.fromarray(im_modify0)
# im_modi0.save(f"{filename}_RGB_M_rescaled.png")

# Create a new array for the new image with the same shape as the original
im_modify = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify[:,:,0] = arr_r_modified+phase_rr_modi_mod
im_modify[:,:,1] = arr_g_modified+phase_gr_modi_mod
im_modify[:,:,2] = arr_b_modified+phase_br_modi_mod

y_offset=center_h-1080//2
im_cropped=im_modify[y_offset:y_offset+1080,:]
im_cropped = im_cropped.astype(np.uint8)
im_modi = Image.fromarray(im_cropped)
im_modi.save(f"{filename}_noNoise_lens.png")
# im_modi.show()

# plt.figure()
# plt.imshow(im_shift_r, cmap='Reds')
# plt.colorbar()
# plt.title("R")
# plt.show()
# plt.figure()
# plt.imshow(im_shift_b, cmap='Blues')
# plt.colorbar()
# plt.title("G")
# plt.show()
# plt.figure()
# plt.imshow(im_shift_g, cmap='Greens')
# plt.colorbar()
# plt.title("B")
# plt.show()