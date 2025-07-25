# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:35:56 2025

@author:Shang Gao 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os
import cv2
import time

start_t=time.time()
os.chdir("C:/Users/Laboratorio/MakeHologram/Con_Cor/")
filename1="L_H0" #flowers_tf  RGB 3circles_exp RGB_500 fl_one
im1=plt.imread(f"{filename1}.png")

filename2="L_V" #flowers_tf  RGB 3circles_exp RGB_500 fl_one
im2=plt.imread(f"{filename2}.png")

def holo(im, f,iterations1, iterations2,power1):
    height=im.shape[0]
    width=im.shape[1]
    """Rescaling is only applied for generating the hologams for experiment."""
    # Define wavelengths (in meters, for rescale)
    # lambda_r = 0.662e-6  # Red wavelength
    # lambda_g = 0.518e-6  # Green wavelength (reference)
    # lambda_b = 0.449e-6  # Blue wavelength
    # # Calculate scaling factors with respect to green
    # scale_r =  lambda_r/lambda_g
    # scale_b =  lambda_b/lambda_g
    
    # h_r, w_r = int(height * scale_r), int(width * scale_r)
    # h_b, w_b = int(height * scale_b), int(width * scale_b)
    # print(h_r,int(h_r), w_r,int(w_r))
    # print(h_b,int(h_b), w_b,int(w_b))
    
    # x_offset_r = (int(w_r)-width) // 2
    # y_offset_r = (int(h_r)-height) // 2
    
    # x_offset_b = (width - int(w_b)) // 2
    # y_offset_b = (height - int(h_b)) // 2
    # # Step 1: Pad the red channel to the scaled size
    # padded_red = np.zeros((h_r, w_r))
    # padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]
    # # Step 2: Crop the blue channel to the scaled size
    # cropped_blue=np.zeros((h_b, w_b))
    # cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]
   
    # padded_B = np.zeros((int(h_r), int(w_r)))
    # padded_B[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 2]
    # # Step 3: Resize both channels back to the original dimensions
    # scaled_red = cv2.resize(padded_red, (width, height), interpolation=cv2.INTER_LINEAR)
    # scaled_blue = cv2.resize(cropped_blue, (width, height), interpolation=cv2.INTER_LINEAR)
    # #<<----------------------End of rescaling preparation-------------------------------->>
    power1=1 # Power control
    # Power controlled intensity for different channels. For the simulation, we only need to change the rescaled red and blue channels to unrescaled channels.
    # #R
    # im_shift_r=fftshift(scaled_red**power1)
    # #G
    # im_shift_g=fftshift(im[:,:,1]**power1)
    # #B
    # im_shift_b=fftshift(scaled_blue**power1)
    #R
    im_shift_r=fftshift(im[:,:,0]**power1)
    #G
    im_shift_g=fftshift(im[:,:,1]**power1)
    #B
    im_shift_b=fftshift(im[:,:,2]**power1)
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
    iterations2=0 # Iteration number for the GS with window.
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
        
    """Lens"""
    lambda_r = 0.662e-6  
    lambda_g = 0.518e-6  
    lambda_b = 0.449e-6  
    arr_r=np.zeros((height, width))
    arr_g=np.zeros((height, width))
    arr_b=np.zeros((height, width))
    pixel_size=4.5e-6
    
    f=-1 # focal length (meters)
    # fg=-0.8
    # fb=-0.6
    center_h=height//2
    center_w=width//2
    # Calculate the phase of lenses according to Eq. (9.1) for three wavelengths.
    for i in range(height):
        for j in range(width):
            r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
            arr_r[i, j] =  r**2 / (f * lambda_r) 
            arr_g[i, j] =  r**2 / (f * lambda_g)
            arr_b[i, j] =  r**2 / (f * lambda_b)
    # # Change the range of the phase into [0,2].
    # arr_r_mod=np.mod(arr_r,2)
    # arr_g_mod=np.mod(arr_g,2)
    # arr_b_mod=np.mod(arr_b,2)
    
    current_field_r=current_field_r*np.exp(1j*arr_r)
    current_field_g=current_field_g*np.exp(1j*arr_g)
    current_field_b=current_field_b*np.exp(1j*arr_b)
    return current_field_r,current_field_g,current_field_b