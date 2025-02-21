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
import time


start_t=time.time()

os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6")
filename="One circle_1024"  
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]

l=300  # Length that is from edge to center.
c_w,c_h=width//2,height//2 # Center
lh,lw=height-2*l,width-2*l  # Height and width for the window.

#R channel
im_shift_r=fftshift(im[:,:,0])
#G channel
im_shift_g=fftshift(im[:,:,1])
#B channel
im_shift_b=fftshift(im[:,:,2])

# Random noise generation.
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi) # Random phase noise
# Apply phase noise to fields, which are square root of intensity in different channels.
im_r_rand=exp_rand*np.sqrt(im_shift_r)

im_g_rand=exp_rand*np.sqrt(im_shift_g)

im_b_rand=exp_rand*np.sqrt(im_shift_b)
# Compute forward 2D Fast Fourier Transform (FFT2) of each channel.
current_field_r =fftshift(fft2(im_r_rand))
current_field_g =fftshift(fft2(im_g_rand))
current_field_b =fftshift(fft2(im_b_rand))
# Iteration1 and iteration2 are for GS algorithm without and width window.
iterations1=40


for j in range(iterations1):
    
    # Reconstrution field that is computed by Inverse FFT2 of the phase value from the previous forward FFT2 computation, and the amplitudes are 1.
    current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
    current_field_g_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_g))))
    current_field_b_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_b))))
    # plt.figure()
    # plt.imshow(fftshift(abs(current_field_r_i)**2),vmax=4e-5, cmap="hot")
    # plt.colorbar()
    # plt.show()
    # Create new reconstrution fields, where the amplitudes are original object fields.
    current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))
    current_field_g_n =np.sqrt(im_shift_g)*np.exp(1j * np.angle(current_field_g_i))
    current_field_b_n =np.sqrt(im_shift_b)*np.exp(1j * np.angle(current_field_b_i))
    
    # Compute Forward FFT2 to obtain target phase holograms.
    current_field_r = fftshift(fft2(current_field_r_n ))
    current_field_g = fftshift(fft2(current_field_g_n ))
    current_field_b = fftshift(fft2(current_field_b_n ))
    # plt.figure()
    # plt.imshow(abs(current_field_r), cmap="hot")
    # plt.colorbar()
    # plt.show()
# Final optimized phase for encoding on SLM, turn angles to grayscales.
optimized_phase_r = np.angle(current_field_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/2)
# plt.figure()
# plt.imshow(abs(current_field_r), cmap="hot")
# plt.colorbar()
# plt.show()
optimized_phase_g = np.angle(current_field_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/2)

optimized_phase_b = np.angle(current_field_b)
phase_br_modi=(optimized_phase_b/np.pi+1)*(255/2)
# Combine three holograms in to one image.
im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi
im_modify_noL[:,:,1] = phase_gr_modi
im_modify_noL[:,:,2] = phase_br_modi
# Crop the final RGB CGHs into different size.
def crop(im_modify,name):
    im_cropped=im_modify
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{filename}_GS_n1_{iterations1}_{name}.png")

C_noL=crop(im_modify_noL, f"nl")

end_t=time.time()
print(f"Time consuming {end_t-start_t}s, iteration {iterations1}")