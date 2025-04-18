# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:45:57 2024

@author: gaosh
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os
import cv2

os.chdir("C:/Users/Laboratorio/MakeHologram/Binarize")
filename="lotus_FH_0"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]

# Define wavelengths (in meters, for rescale)
lambda_r = 0.680e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.461e-6  # Blue wavelength
# Calculate scaling factors with respect to green
scale_r =  lambda_r/lambda_g
scale_b =  lambda_b/lambda_g

h_r, w_r = int(height * scale_r), int(width * scale_r)
h_b, w_b = int(height * scale_b), int(width * scale_b)
print(h_r,int(h_r), w_r,int(w_r))
print(h_b,int(h_b), w_b,int(w_b))

x_offset_r = (int(w_r)-width) // 2
y_offset_r = (int(h_r)-height) // 2
print(x_offset_r,y_offset_r)
x_offset_b = (width - int(w_b)) // 2
y_offset_b = (height - int(h_b)) // 2
print(x_offset_b,y_offset_b)
# Step 1: Pad the red channel to the scaled size
padded_red = np.zeros((h_r, w_r))
padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]

# Step 2: Crop the blue channel to the scaled size
cropped_blue=np.zeros((h_b, w_b))
cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]

# Step 3: Resize both channels back to the original dimensions
scaled_red = cv2.resize(padded_red, (width, height), interpolation=cv2.INTER_LINEAR)
scaled_blue = cv2.resize(cropped_blue, (width, height), interpolation=cv2.INTER_LINEAR)

im_manipulated=np.zeros((height,width,3))
im_manipulated[:,:,0]=scaled_red
im_manipulated[:,:,1]=im[:,:,1]
im_manipulated[:,:,2]=scaled_blue

#R
im_shift_r=fftshift(scaled_red)
#G
im_shift_g=fftshift(im[:,:,1])
#B
im_shift_b=fftshift(scaled_blue)

# random
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
#R
im_r_rand=exp_rand*im_shift_r

im_g_rand=exp_rand*im_shift_g

im_b_rand=exp_rand*im_shift_b

current_field_r = fftshift(fft2(im_r_rand ))
current_field_g =fftshift(fft2(im_g_rand))
current_field_b =fftshift(fft2(im_b_rand))

iterations=1
for i in range(iterations):
    
    # Inverse Fourier Transform to initial plane
    current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
    current_field_g_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_g))))
    current_field_b_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_b))))
    
    target_phase_constraint = (np.angle(current_field_r_i) +
                           np.angle(current_field_g_i) +
                           np.angle(current_field_b_i)) / 3
    current_field_r_n = im_shift_r *np.exp(1j *target_phase_constraint)#*exp_rand# 
    current_field_g_n = im_shift_g *np.exp(1j *target_phase_constraint)#*exp_rand#  
    current_field_b_n = im_shift_b *np.exp(1j *target_phase_constraint)#*exp_rand# 
    
    # Forward Fourier Transform to the target plane
    current_field_r = fftshift(fft2(current_field_r_n ))
    current_field_g = fftshift(fft2(current_field_g_n ))
    current_field_b = fftshift(fft2(current_field_b_n ))

# plt.figure()
# plt.imshow(target_phase_constraint,cmap="gray")
# plt.title("average phase")
# plt.colorbar()
# plt.show()

# Final optimized phase for display or application on SLM
optimized_phase_r_o = np.angle(current_field_r)
optimized_phase_r = np.where(optimized_phase_r_o < 0, 0, 1)
phase_rr_modi=optimized_phase_r*184#(optimized_phase_r/np.pi+1)*(255/1.85)
phase_rr_modi_mod=np.mod(phase_rr_modi,255)

optimized_phase_g_o = np.angle(current_field_g)
optimized_phase_g = np.where(optimized_phase_g_o < 0, 0, 1)
phase_gr_modi=optimized_phase_g*136#(optimized_phase_g/np.pi+1)*(255/2.63)
phase_gr_modi_mod=np.mod(phase_gr_modi,255)

optimized_phase_b_o = np.angle(current_field_b)
optimized_phase_b = np.where(optimized_phase_b_o < 0, 0, 1)
phase_br_modi=optimized_phase_b*112#(optimized_phase_b/np.pi+1)*(255/3.55)
phase_br_modi_mod=np.mod(phase_br_modi,255)

# plt.figure()
# plt.imshow(phase_rr_modi,cmap="Reds")
# plt.title("R")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(optimized_phase_r,cmap="Reds")
# plt.title("R")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(phase_gr_modi,cmap="Greens")
# plt.title("G")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(optimized_phase_g,cmap="Greens")
# plt.title("G")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(phase_br_modi,cmap="Blues")
# plt.title("B")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(optimized_phase_b,cmap="Blues")
# plt.title("B")
# plt.colorbar()
# plt.show()


"""Lens"""
lambda_r = 0.633e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.450e-6  
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
# Create a new array for the new image with the same shape as the original
im_modify_c = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_c[:,:,0] = phase_rr_modi_mod+arr_r_modified
im_modify_c[:,:,1] = phase_gr_modi_mod+arr_g_modified
im_modify_c[:,:,2] = phase_br_modi_mod+arr_b_modified

im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi_mod
im_modify_noL[:,:,1] = phase_gr_modi_mod
im_modify_noL[:,:,2] = phase_br_modi_mod

def crop(im_modify,name):
    y_offset=center_h-1080//2
    im_cropped=im_modify[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"Bi_{filename}_184_136_112{iterations}_C_{name}.png")
# R=crop(im_modify_r, "r")
# G=crop(im_modify_g, "g")
# B=crop(im_modify_b, "b")
C=crop(im_modify_c, "L")
C_noL=crop(im_modify_noL, "nL")