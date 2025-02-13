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
os.chdir("C:/Users/Laboratorio/MakeHologram/Wirtingers_Holography")
filename="RGB_500" 
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]
"""Rescaling is only applied for generating the hologams for experiment."""
# Define wavelengths (in meters, for rescale)
lambda_r = 0.662e-6  # Red wavelength
lambda_g = 0.518e-6  # Green wavelength (reference)
lambda_b = 0.449e-6  # Blue wavelength
# Calculate scaling factors with respect to green
scale_r =  lambda_r/lambda_g
scale_b =  lambda_b/lambda_g

h_r, w_r = int(height * scale_r), int(width * scale_r)
h_b, w_b = int(height * scale_b), int(width * scale_b)
print(h_r,int(h_r), w_r,int(w_r))
print(h_b,int(h_b), w_b,int(w_b))

x_offset_r = (int(w_r)-width) // 2
y_offset_r = (int(h_r)-height) // 2

x_offset_b = (width - int(w_b)) // 2
y_offset_b = (height - int(h_b)) // 2
# Step 1: Pad the red channel to the scaled size
padded_red = np.zeros((h_r, w_r))
padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]
# Step 2: Crop the blue channel to the scaled size
cropped_blue=np.zeros((h_b, w_b))
cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]
t=1
padded_B = np.zeros((int(t*h_r), int(t*w_r)))
padded_B[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 2]
# Step 3: Resize both channels back to the original dimensions
scaled_red = cv2.resize(padded_red, (width, height), interpolation=cv2.INTER_LINEAR)
scaled_blue = cv2.resize(cropped_blue, (width, height), interpolation=cv2.INTER_LINEAR)
#<<----------------------End of rescaling preparation-------------------------------->>
power1=4 # Power control
# Power controlled intensity for different channels. For the simulation, we only need to change the rescaled red and blue channels to unrescaled channels.
#R
im_shift_r=fftshift(scaled_red**power1)
#G
im_shift_g=fftshift(im[:,:,1]**power1)
#B
im_shift_b=fftshift(scaled_blue**power1)
# Genaration of random phase noise.
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
# Multiply noise with the field of the preprocessed orignal objects.
im_r_rand=exp_rand*np.sqrt(im_shift_r)

im_g_rand=exp_rand*np.sqrt(im_shift_g)

im_b_rand=exp_rand*np.sqrt(im_shift_b)
# Forward FFT of different channels
current_field_r =fftshift(fft2(im_r_rand ))
current_field_g =fftshift(fft2(im_g_rand))
current_field_b =fftshift(fft2(im_b_rand))

iterations1=20 # Iteration number for the GS without window.
iterations2=20 # Iteration number for the GS with window.

# Loop for the GS without window.
for j in range(iterations1): 
    # Reconstrution field that is computed by Inverse FFT2 of the phase value from the previous forward FFT2 computation, and the amplitudes are 1.
    current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
    current_field_g_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_g))))
    current_field_b_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_b))))  
    # Create new reconstrution fields, where the amplitudes are original object fields.
    current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))
    current_field_g_n =np.sqrt(im_shift_g)*np.exp(1j * np.angle(current_field_g_i))
    current_field_b_n =np.sqrt(im_shift_b)*np.exp(1j * np.angle(current_field_b_i))
    # Compute Forward FFT2 to obtain target phase holograms.
    current_field_r = fftshift(fft2(current_field_r_n ))
    current_field_g = fftshift(fft2(current_field_g_n ))
    current_field_b = fftshift(fft2(current_field_b_n ))
# Loop for the GS with window.    
for i in range(iterations2): 
    # Inverse FFT2 computation that is the same as the GS without window loop.
    current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
    current_field_g_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_g))))
    current_field_b_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_b))))
    # Retrieve both amplitudes and phase of inverse FFT2 computation, and normalize amplitude part by the maximum in each channel.
    current_field_r_i_t =np.sqrt(abs(current_field_r_i)**2/np.max(abs(current_field_r_i)**2)) *np.exp(1j * np.angle(current_field_r_i))
    current_field_g_i_t = np.sqrt(abs(current_field_g_i)**2/np.max(abs(current_field_g_i)**2)) *np.exp(1j * np.angle(current_field_g_i))
    current_field_b_i_t =np.sqrt(abs(current_field_b_i)**2/np.max(abs(current_field_b_i)**2)) *np.exp(1j * np.angle(current_field_b_i))
    # Create new object fields with original fields, and the phases are the same as the previous step.
    current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))#*exp_rand
    current_field_g_n =np.sqrt(im_shift_g)*np.exp(1j * np.angle(current_field_g_i))#*exp_rand
    current_field_b_n =np.sqrt(im_shift_b)*np.exp(1j * np.angle(current_field_b_i))#*exp_rand
    
    l=620 # Define widow size, which is the length from window edge to image edge.
    c_w,c_h=width//2,height//2
    # For experiment, rescale the window size for red and blue channels.
    lr=int((height*(scale_r-1)+2*l)/(2*scale_r))
    lb=int((height*(scale_b-1)+2*l)/(2*scale_b))
    # In the window, the amplitudes are from original. Otherwise, the amplitudes are from retrieved amplitudes in inverse FFT2 computation.
    current_field_r_n[:,c_w-lr:c_w+lr]=current_field_r_i_t[:,c_w-lr:c_w+lr]
    current_field_g_n[:,c_w-l:c_w+l]=current_field_g_i_t[:,c_w-l:c_w+l]
    current_field_b_n[:,c_w-lb:c_w+lb]=current_field_b_i_t[:,c_w-lb:c_w+lb]
    
    current_field_r_n[c_h-lr:c_h+lr,0:c_w-lr]=current_field_r_i_t[c_h-lr:c_h+lr,0:c_w-lr]
    current_field_g_n[c_h-l:c_h+l,0:c_w-l]=current_field_g_i_t[c_h-l:c_h+l,0:c_w-l]
    current_field_b_n[c_h-lb:c_h+lb,0:c_w-lb]=current_field_b_i_t[c_h-lb:c_h+lb,0:c_w-lb]
    
    current_field_r_n[c_h-lr:c_h+lr,c_w+lr:width]=current_field_r_i_t[c_h-lr:c_h+lr,c_w+lr:width]
    current_field_g_n[c_h-l:c_h+l,c_w+l:width]=current_field_g_i_t[c_h-l:c_h+l,c_w+l:width]
    current_field_b_n[c_h-lb:c_h+lb,c_w+lb:width]=current_field_b_i_t[c_h-lb:c_h+lb,c_w+lb:width]    
    # Compute Forward FFT2 to obtain phase holograms with window.
    current_field_r = fftshift(fft2(current_field_r_n ))
    current_field_g = fftshift(fft2(current_field_g_n ))
    current_field_b = fftshift(fft2(current_field_b_n ))

# Phase modulation depth for experiment, and they are all 2 for simulation.
rdp,gdp,bdp=1.78,2.56,3.5
# Final optimized phase for encoding on SLM, turn angles to grayscales.
optimized_phase_r = np.angle(current_field_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/rdp)

optimized_phase_g = np.angle(current_field_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/gdp)

optimized_phase_b = np.angle(current_field_b)
phase_br_modi=(optimized_phase_b/np.pi+1)*(255/bdp)

"""Lens"""
lambda_r = 0.662e-6  
lambda_g = 0.518e-6  
lambda_b = 0.449e-6  
arr_r=np.zeros((height, width))
arr_g=np.zeros((height, width))
arr_b=np.zeros((height, width))
pixel_size=4.5e-6
f=-2 # focal length (meters).
center_h=height//2
center_w=width//2
# Calculate the phase of lenses according to Eq. (9.1) for three wavelengths.
for i in range(height):
    for j in range(width):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        arr_r[i, j] =  r**2 / (f * lambda_r) 
        arr_g[i, j] =  r**2 / (f * lambda_g)
        arr_b[i, j] =  r**2 / (f * lambda_b)
# Change the range of the phase into [0,2].
arr_r_mod=np.mod(arr_r,2)
arr_g_mod=np.mod(arr_g,2)
arr_b_mod=np.mod(arr_b,2)
# Map the phase to grayscale by corresponding phase modulation.
arr_r_modified=arr_r_mod*(255/rdp)
arr_g_modified=arr_g_mod*(255/gdp)
arr_b_modified=arr_b_mod*(255/bdp)
# Save and crop the corresponding RGB CGHs with/without lens for experiment and simulation, respectively.
im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi
im_modify_noL[:,:,1] = phase_gr_modi
im_modify_noL[:,:,2] = phase_br_modi

im_modify_c = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_c[:,:,0] = phase_rr_modi+arr_r_modified
im_modify_c[:,:,1] = phase_gr_modi+arr_g_modified
im_modify_c[:,:,2] = phase_br_modi+arr_b_modified

def crop(im_modify,name):
    y_offset=center_h-1080//2
    im_cropped=im_modify[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{filename}_GS_It1_{iterations1}_It2_{iterations2}_{name}_p {power1}.png")

C=crop(im_modify_c, "L")
C_noL=crop(im_modify_noL, "nL")

end_t=time.time()
print(f"Time consuming {end_t-start_t}s, iteration {iterations1+iterations2}")