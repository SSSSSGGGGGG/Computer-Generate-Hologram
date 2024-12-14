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

os.chdir("C:/Users/gaosh/Documents/python/Computer-Generate-Hologram/Wirting_Holography")
filename="3D_dear_fit_sh"
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
# plt.imsave(f"scaled {filename}.png", im_manipulated)
# plt.figure()
# plt.imshow(im_manipulated)
# # plt.colorbar()
# plt.title("RGB")
# plt.show()
#R
im_shift_r=fftshift(scaled_red)
#G
im_shift_g=fftshift(im[:,:,1])
#B
im_shift_b=fftshift(scaled_blue)

im_re = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_re[:,:,0]=fftshift(scaled_red)
im_re[:,:,1]=fftshift(im[:,:,1])
im_re[:,:,2]=fftshift(scaled_blue)

# use W gradient to improve phase
def wirtinger_phase_improve(intensity,iterations,lr,verbose=False):
    
    intensity_r=intensity[:,:,0]
    intensity_g=intensity[:,:,1]
    intensity_b=intensity[:,:,2]
    size=intensity_r.shape
    # generate ramdom magnitude and phase
    reconstruted_field_r=np.random.uniform(0.5,1.0,size)*np.exp(1j*np.random.uniform(0,2*np.pi,size))
    reconstruted_field_g=np.random.uniform(0.5,1.0,size)*np.exp(1j*np.random.uniform(0,2*np.pi,size))
    reconstruted_field_b=np.random.uniform(0.5,1.0,size)*np.exp(1j*np.random.uniform(0,2*np.pi,size))    
    for it in range (iterations):
        # calculate intensity of the mimiced
        current_in_r=np.abs(reconstruted_field_r)**2
        current_in_g=np.abs(reconstruted_field_g)**2
        current_in_b=np.abs(reconstruted_field_b)**2
        # define a loss function: mean square error
        loss = (np.mean((current_in_r - intensity_r)**2) +np.mean((current_in_g - intensity_g)**2) +np.mean((current_in_b - intensity_b)**2))

        
        # compute gradient, this is the gradient of loss respective to imaginary field
        grad_r=(current_in_r-intensity_r)*reconstruted_field_r/np.abs(reconstruted_field_r+1e-8)
        reconstruted_field_r-=lr*grad_r
        reconstruted_field_r=np.sqrt(intensity_r)*np.exp(1j*np.angle(reconstruted_field_r))
        
        grad_g=(current_in_g-intensity_g)*reconstruted_field_g/np.abs(reconstruted_field_g+1e-8)
        reconstruted_field_g-=lr*grad_g
        reconstruted_field_g=np.sqrt(intensity_g)*np.exp(1j*np.angle(reconstruted_field_g))
    
        grad_b=(current_in_b-intensity_b)*reconstruted_field_b/np.abs(reconstruted_field_b+1e-8)
        reconstruted_field_b-=lr*grad_b
        reconstruted_field_b=np.sqrt(intensity_b)*np.exp(1j*np.angle(reconstruted_field_b))
    
    return reconstruted_field_r,reconstruted_field_g,reconstruted_field_b

# measurd_in_r=np.abs(im_shift_r)**2
measurd_in=np.abs(im_re)**2
measurd_in_FT_r=fftshift(fft2(im_shift_r))
phase_in_r=np.angle(measurd_in_FT_r)

iterations=10
final_field_r,final_field_g,final_field_b=wirtinger_phase_improve(measurd_in, iterations, 0.01)
final_field_FT_r=fftshift(fft2(final_field_r))
# Final optimized phase for display or application on SLM
optimized_phase_r = np.angle(final_field_FT_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/1.85)
# phase_rr_modi_mod=np.mod(phase_rr_modi,255)
final_field_FT_g=fftshift(fft2(final_field_g))
optimized_phase_g = np.angle(final_field_FT_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/2.63)
# phase_gr_modi_mod=np.mod(phase_gr_modi,255)
final_field_FT_b=fftshift(fft2(final_field_b))
optimized_phase_b = np.angle(final_field_FT_b)
phase_br_modi=(optimized_phase_b/np.pi+1)*(255/3.55)
# phase_br_modi_mod=np.mod(phase_br_modi,255)

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

im_modify_r = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_g = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_b = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_r_nl = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_g_nl = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_b_nl = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))

# Fill each channel with respective modifications
im_modify_r[:,:,0] = phase_rr_modi + arr_r_modified
im_modify_g[:,:,1]= phase_gr_modi+ arr_g_modified
im_modify_b[:,:,2] = phase_br_modi + arr_b_modified

im_modify_r_nl[:,:,0] = phase_rr_modi 
im_modify_g_nl[:,:,1]= phase_gr_modi
im_modify_b_nl[:,:,2] = phase_br_modi
# Create a new array for the new image with the same shape as the original
im_modify_c = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_c[:,:,0] = phase_rr_modi+arr_r_modified
im_modify_c[:,:,1] = phase_gr_modi+arr_g_modified
im_modify_c[:,:,2] = phase_br_modi+arr_b_modified

im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi
im_modify_noL[:,:,1] = phase_gr_modi
im_modify_noL[:,:,2] = phase_br_modi
def crop(im_modify,name):
    y_offset=center_h-1080//2
    im_cropped=im_modify[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{filename}_{iterations}_C_{name}.png")
# R=crop(im_modify_r, "L")
R_noL=crop(im_modify_r_nl, "noL")
# G=crop(im_modify_g, "L")
# G_noL=crop(im_modify_g_nl, "noL")
# B=crop(im_modify_b, "L")
# B_noL=crop(im_modify_b_nl, "noL")
# C=crop(im_modify_c, "L")
# C_noL=crop(im_modify_noL, "nL")