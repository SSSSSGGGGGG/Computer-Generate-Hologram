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
os.chdir("C:/Users/gaosh/Documents/python/Digital-hologram/FFT of imgs")
filename="planets"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]
#R
im_shift_r=fftshift(im[:,:,0])
#G
im_shift_g=fftshift(im[:,:,1])
#B
im_shift_b=fftshift(im[:,:,2])

# random
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
#R
im_r_rand=exp_rand*im_shift_r
im_rr_ft=fftshift(fft2(im_r_rand))
phase_rr = np.angle(im_rr_ft)
phase_rr_new=phase_rr.astype(np.uint8)
phase_rr_save=Image.fromarray(phase_rr_new)

im_g_rand=exp_rand*im_shift_g
im_gr_ft=fftshift(fft2(im_g_rand))
phase_gr = np.angle(im_gr_ft)
phase_gr_new=phase_gr.astype(np.uint8)
phase_gr_save=Image.fromarray(phase_gr_new)

im_b_rand=exp_rand*im_shift_b
im_br_ft=fftshift(fft2(im_b_rand))
phase_br = np.angle(im_br_ft)
phase_br_new=phase_br.astype(np.uint8)
phase_br_save=Image.fromarray(phase_br_new)
# Start with the initial guess in the target plane
current_field_r = im_r_rand
current_field_g =im_g_rand
current_field_b =im_b_rand
iterations=10
for i in range(iterations):
    # Forward Fourier Transform to the target plane
    field_target_r = fftshift(fft2(current_field_r ))
    field_target_g = fftshift(fft2(current_field_g ))
    field_target_b = fftshift(fft2(current_field_b ))
    # Inverse Fourier Transform to initial plane
    current_field_r = ifft2(ifftshift(np.exp(1j * np.angle(field_target_r))))
    current_field_g = ifft2(ifftshift(np.exp(1j * np.angle(field_target_g))))
    current_field_b = ifft2(ifftshift(np.exp(1j * np.angle(field_target_b))))
    # Impose constraints in the initial plane (e.g., unit amplitude or custom profile)
    current_field_r = exp_rand*im_shift_r*np.exp(1j * np.angle(current_field_r))
    current_field_g = exp_rand*im_shift_g*np.exp(1j * np.angle(current_field_g))
    current_field_b = exp_rand*im_shift_b*np.exp(1j * np.angle(current_field_b))
# Final optimized phase for display or application on SLM
optimized_phase_r = np.angle(current_field_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/1.85)
phase_rr_modi_mod=np.mod(phase_rr_modi,255)

optimized_phase_g = np.angle(current_field_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/2.63)
phase_gr_modi_mod=np.mod(phase_gr_modi,255)

optimized_phase_g = np.angle(current_field_g)
phase_br_modi=(optimized_phase_g+1)*(255/3.55)
phase_br_modi_mod=np.mod(phase_br_modi,255)

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
im_modify = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify[:,:,0] = phase_rr_modi_mod+arr_r_modified
im_modify[:,:,1] = phase_gr_modi_mod+arr_g_modified
im_modify[:,:,2] = phase_br_modi_mod+arr_b_modified
im_modify = im_modify.astype(np.uint8)
im_modi = Image.fromarray(im_modify)
im_modi.save(f"{filename}_GS {iterations}_lens.png")