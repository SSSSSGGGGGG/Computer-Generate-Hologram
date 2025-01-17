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
import time

start_t=time.time()
os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis")
filename="flowers_960"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]

# Define wavelengths (in meters, for rescale)
lambda_r = 0.680e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.461e-6  # Blue wavelength

#R
im_shift_r=fftshift(im[:,:,0])
#G
im_shift_g=fftshift(im[:,:,1])
#B
im_shift_b=fftshift(im[:,:,2])

im_re = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_re[:,:,0]=fftshift(im[:,:,0])
im_re[:,:,1]=fftshift(im[:,:,1])
im_re[:,:,2]=fftshift(im[:,:,2])
power2=1
power3=1
iterations=10 #!!!!!!
# use W gradient to improve phase
def wirtinger_phase_improve(intensity,iterations,lr):
    
    intensity_r=intensity[:,:,0]  # this unit is 2 times the reconstruted_field
    intensity_g=intensity[:,:,1]
    intensity_b=intensity[:,:,2]
    size=intensity_r.shape
    # generate ramdom magnitude and phase
    reconstruted_field_r=intensity_r**power2*np.exp(1j*np.random.uniform(0,2*np.pi,size))
    reconstruted_field_g=intensity_g**power2*np.exp(1j*np.random.uniform(0,2*np.pi,size))
    reconstruted_field_b=intensity_b**power2*np.exp(1j*np.random.uniform(0,2*np.pi,size))    
    for it in range (iterations):
        # calculate intensity of the mimiced
        current_in_r=np.abs(reconstruted_field_r)**2# power was here, since i used power so power=0.5, then the power of 2 should be back.
        current_in_g=np.abs(reconstruted_field_g)**2
        current_in_b=np.abs(reconstruted_field_b)**2
        loss = (np.mean((current_in_r - intensity_r)**2) +np.mean((current_in_g - intensity_g)**2) +np.mean((current_in_b - intensity_b)**2))

        
        # compute gradient, this is the gradient of loss respective to imaginary field
        grad_r=(current_in_r-intensity_r)*reconstruted_field_r/np.abs(reconstruted_field_r+1e-8)
        reconstruted_field_r-=lr*grad_r
        reconstruted_field_r=intensity_r**power3*np.exp(1j*np.angle(reconstruted_field_r))
        
        grad_g=(current_in_g-intensity_g)*reconstruted_field_g/np.abs(reconstruted_field_g+1e-8)
        reconstruted_field_g-=lr*grad_g
        reconstruted_field_g=intensity_g**power3*np.exp(1j*np.angle(reconstruted_field_g))
    
        grad_b=(current_in_b-intensity_b)*reconstruted_field_b/np.abs(reconstruted_field_b+1e-8)
        reconstruted_field_b-=lr*grad_b
        reconstruted_field_b=intensity_b**power3*np.exp(1j*np.angle(reconstruted_field_b))
        power2 * np.exp(1j * np.angle(reconstruted_field_b))

    
    return reconstruted_field_r,reconstruted_field_g,reconstruted_field_b
power1=1
# measurd_in_r=np.abs(im_shift_r)**2
measurd_in=im_re**power1
measurd_in_FT_r=fftshift(fft2(im_shift_r))
phase_in_r=np.angle(measurd_in_FT_r)


final_field_r,final_field_g,final_field_b=wirtinger_phase_improve(measurd_in, iterations, 0.01)
final_field_FT_r=fftshift(fft2(final_field_r))
mag_final_field_FT_r=np.abs(final_field_r)
# Final optimized phase for display or application on SLM
optimized_phase_r = np.angle(final_field_FT_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/2)
# phase_rr_modi_mod=np.mod(phase_rr_modi,255)
final_field_FT_g=fftshift(fft2(final_field_g))
optimized_phase_g = np.angle(final_field_FT_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/2)
# phase_gr_modi_mod=np.mod(phase_gr_modi,255)
final_field_FT_b=fftshift(fft2(final_field_b))
optimized_phase_b = np.angle(final_field_FT_b)
phase_br_modi=(optimized_phase_b/np.pi+1)*(255/2)
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
# # Create a new array for the new image with the same shape as the original
im_modify_c = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_c[:,:,0] = phase_rr_modi+arr_r_modified
im_modify_c[:,:,1] = phase_gr_modi+arr_g_modified
im_modify_c[:,:,2] = phase_br_modi+arr_b_modified

im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi
im_modify_noL[:,:,1] = phase_gr_modi
im_modify_noL[:,:,2] = phase_br_modi
def crop(im_modify,name):
    # y_offset=center_h-1080//2
    im_cropped=im_modify#◘[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{filename}_wh_{iterations,name}_p3_{power3}_input{power1}.png")
# R=crop(im_modify_r, "L")
# R_noL=crop(im_modify_r_nl, "noL")
# G=crop(im_modify_g, "L")
# G_noL=crop(im_modify_g_nl, "noL")
# B=crop(im_modify_b, "L")
# B_noL=crop(im_modify_b_nl, "noL")
# C=crop(im_modify_c, "nL")
C_noL=crop(im_modify_noL, "nL")
end_t=time.time()
print(f"Time consuming {end_t-start_t}s, iteration {iterations}")