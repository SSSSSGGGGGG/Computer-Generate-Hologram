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
filename="RGB100p"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]

# Define wavelengths (in meters, for example)
lambda_r = 0.633e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.450e-6  # Blue wavelength
# Calculate scaling factors with respect to green
scale_r = lambda_g / lambda_r
scale_b = lambda_g / lambda_b

h_r=height*scale_r
w_r=width*scale_r
h_b=height*scale_b
w_b=width*scale_b

scaled_red = im[:,:,0]
scaled_blue = im[:,:,2]

#R
im_shift_r=fftshift(im[:,:,0])
#G
im_shift_g=fftshift(im[:,:,1])
#B
im_shift_b=fftshift(im[:,:,2])

# random

def rand(height, width):
    rand=np.random.uniform(0, 1, (height, width))
    rand_2pi=np.pi*rand
    rand_ma=np.max(rand_2pi)
    rand_mi=np.min(rand_2pi)
    exp_rand=np.exp(1j*rand_2pi)
    return exp_rand
    
#R
im_r_rand=rand(int(w_r), int(h_r))*scaled_red
im_rr_ft=fftshift(fft2(im_r_rand))
phase_rr = np.angle(im_rr_ft)
phase_rr_new=phase_rr.astype(np.uint8)
phase_rr_save=Image.fromarray(phase_rr_new)
phase_rr_modi=(np.angle(im_rr_ft)/np.pi+1)*(255/1.85)
phase_rr_modi_mod=np.mod(phase_rr_modi,255)
# R_blaze=plt.imread("C:/Users/Laboratorio/MakeHologram/tri_blue is center/R_bl2_p200.png")
# cropped_image = R_blaze[0:height, 0:width]
# phase_rr_bl=phase_rr_modi_mod+cropped_image[:,:,0]*255
# phase_rr_bl_mod=np.mod(phase_rr_bl,255)

#G
im_g_rand=rand(height, width)*im_shift_g
im_gr_ft=fftshift(fft2(im_g_rand))
phase_gr = np.angle(im_gr_ft)
phase_gr_new=phase_gr.astype(np.uint8)
phase_gr_save=Image.fromarray(phase_gr_new)
phase_gr_modi=(np.angle(im_gr_ft)/np.pi+1)*(255/2.63)
phase_gr_modi_mod=np.mod(phase_gr_modi,255)
# G_blaze=plt.imread("C:/Users/Laboratorio/MakeHologram/tri_blue is center/G_bl2_p300.png")
# cropped_imageG = G_blaze[0:height, 0:width]
# phase_gr_bl=phase_gr_modi_mod+cropped_imageG[:,:,1]*255

#B
im_b_rand=rand(int(w_b), int(h_b))*scaled_blue
im_br_ft=fftshift(fft2(im_b_rand))
phase_br = np.angle(im_br_ft)
phase_br_new=phase_br.astype(np.uint8)
phase_br_save=Image.fromarray(phase_br_new)
phase_br_modi=(np.angle(im_br_ft)/np.pi+1)*(255/3.55)
phase_br_modi_mod=np.mod(phase_br_modi,255)

# # Create a new array for the new image with the same shape as the original
# im_new_array = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
# # Modify the new array
# im_new_array[:,:,0] = phase_rr_modi_mod
# im_new_array[:,:,1] = phase_gr_modi_mod
# im_new_array[:,:,2] = phase_br_modi_mod
# im_new_array = im_new_array.astype(np.uint8)
# im_new = Image.fromarray(im_new_array)
# im_new.save(f"{filename}_RGB_rand_M.png")


padded_phase_red = np.zeros((1024, 1024), dtype=phase_rr_modi_mod.dtype)
padded_phase_blue = np.zeros((1024, 1024), dtype=phase_rr_modi_mod.dtype)
# Calculate the offsets to center the 860x860 array within 1024x1024
x_offset_r = (1024 - int(w_r)) // 2
y_offset_r = (1024 - int(h_r)) // 2

x_offset_b = (int(w_b)-1024) // 2
y_offset_b = (int(h_b)-1024) // 2

padded_phase_red[y_offset_r:y_offset_r + int(h_r), x_offset_r:x_offset_r + int(w_r)] = phase_rr_modi_mod
padded_phase_blue =phase_br_modi_mod[y_offset_b:y_offset_b + height, x_offset_b:x_offset_b + width]

print(y_offset_r,y_offset_r + int(h_r), x_offset_r,x_offset_r + int(w_r))
print(y_offset_b,y_offset_b + height, x_offset_b,x_offset_b + width)
# Create a new array for the new image with the same shape as the original
im_modify = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
# Modify the new array
im_modify[:,:,0] = padded_phase_red
im_modify[:,:,1] = phase_gr_modi_mod
im_modify[:,:,2] = padded_phase_blue 
im_modify = im_modify.astype(np.uint8)
im_modi = Image.fromarray(im_modify)
im_modi.save(f"{filename}_RGB_rand_M_scaled.png")
# im_modi.show()

# # Create a new array for the new image with the same shape as the original
# im_modify_bl = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
# # Modify the new array
# im_modify_bl[:,:,0] = phase_rr_bl
# im_modify_bl[:,:,1] = phase_gr_bl
# im_modify_bl[:,:,2] = phase_br_modi_mod
# im_modify_bl = im_modify_bl.astype(np.uint8)
# im_modi_bl = Image.fromarray(im_modify_bl)
# im_modi_bl.save(f"{filename}_RGB_randModi_bl.png")
# # im_modi_bl.show()

plt.figure()
plt.imshow(im_shift_r, cmap='Reds')
plt.colorbar()
plt.title("R")
plt.show()
plt.figure()
plt.imshow(scaled_red, cmap='Reds')
plt.colorbar()
plt.title("R")
plt.show()
# plt.figure(5)
# plt.imshow(im_shift_g, cmap='Greens')
# plt.colorbar()
# plt.title("G")
# plt.show()

# plt.figure()
# plt.imshow(phase_br_modi_mod, cmap='Blues')
# plt.colorbar()
# plt.title("B")
# plt.show()
# plt.figure()
# plt.imshow(scaled_phase_blue, cmap='Blues')
# plt.colorbar()
# plt.title("B")
# plt.show()
# plt.figure()
# plt.imshow(padded_phase_blue, cmap='Blues')
# plt.colorbar()
# plt.title("B")
# plt.show()
# # plt.imsave(f"{filename}_m_rand.png", phase_r,cmap='gray')