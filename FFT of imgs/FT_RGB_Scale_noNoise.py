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
filename="t"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]

# Define wavelengths (in meters, for example)
lambda_r = 0.633e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.450e-6  # Blue wavelength
# # Calculate scaling factors with respect to green
# scale_r =  lambda_r/lambda_g
# scale_b =  lambda_b/lambda_g

# h_r, w_r = int(height * scale_r), int(width * scale_r)
# h_b, w_b = int(height * scale_b), int(width * scale_b)
# print(h_r,int(h_r), w_r,int(w_r))
# print(h_b,int(h_b), w_b,int(w_b))

# x_offset_r = (int(w_r)-width) // 2
# y_offset_r = (int(h_r)-height) // 2
# print(x_offset_r,y_offset_r)
# x_offset_b = (width - int(w_b)) // 2
# y_offset_b = (height - int(h_b)) // 2
# print(x_offset_b,y_offset_b)
# # Step 1: Pad the red channel to the scaled size
# padded_red = np.zeros((h_r, w_r))
# padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]

# # Step 2: Crop the blue channel to the scaled size
# cropped_blue=np.zeros((h_b, w_b))
# cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]

# # Step 3: Resize both channels back to the original dimensions
# scaled_red = cv2.resize(padded_red, (width, height), interpolation=cv2.INTER_LINEAR)
# scaled_blue = cv2.resize(cropped_blue, (width, height), interpolation=cv2.INTER_LINEAR)

# im_manipulated=np.zeros((height,width,3))
# im_manipulated[:,:,0]=im[:,:,0]
# im_manipulated[:,:,1]=im[:,:,1]
# im_manipulated[:,:,2]=im[:,:,2]
# plt.imsave(f"scaled {filename}.png", im_manipulated)
# plt.figure()
# plt.imshow(im_manipulated)
# # plt.colorbar()
# plt.title("RGB")
# plt.show()
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
# phase_rr_new=phase_rr.astype(np.uint8)
# phase_rr_save=Image.fromarray(phase_rr_new)
phase_rr_modi=(np.angle(im_rr_ft)/np.pi+1)*(255/1.85)
phase_rr_modi_mod=np.mod(phase_rr_modi,255)

#G
im_g_rand=im_shift_g
im_gr_ft=fftshift(fft2(im_g_rand))
phase_gr = np.angle(im_gr_ft)
# phase_gr_new=phase_gr.astype(np.uint8)
# phase_gr_save=Image.fromarray(phase_gr_new)
phase_gr_modi=(np.angle(im_gr_ft)/np.pi+1)*(255/2.63)
phase_gr_modi_mod=np.mod(phase_gr_modi,255)

#B
im_b_rand=im_shift_b
im_br_ft=fftshift(fft2(im_b_rand))
phase_br = np.angle(im_br_ft)
# phase_br_new=phase_br.astype(np.uint8)
# phase_br_save=Image.fromarray(phase_br_new)
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
im_modi.save(f"{filename}_RGB_M_rescaled_lens.png")
# im_modi.show()

plt.figure()
plt.imshow(im[:,:,0], cmap='Reds')
plt.colorbar()
plt.title("R")
plt.show()
plt.figure()
plt.imshow(im[:,:,2], cmap='Blues')
plt.colorbar()
plt.title("G")
plt.show()
plt.figure()
plt.imshow(im[:,:,1], cmap='Greens')
plt.colorbar()
plt.title("B")
plt.show()