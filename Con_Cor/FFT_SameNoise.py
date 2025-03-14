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
# os.chdir("C:/Users/Laboratorio/MakeHologram/Wirtingers_Holography")
# filename="RGB_500" 
# im=plt.imread(f"{filename}.png")

height=1920#im.shape[0]
width=1920#im.shape[1]

center_h=height//2
center_w=width//2
# Genaration of random phase noise.
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)


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
# print(h_r,int(h_r), w_r,int(w_r))
# print(h_b,int(h_b), w_b,int(w_b))

x_offset_r = (int(w_r)-width) // 2
y_offset_r = (int(h_r)-height) // 2

x_offset_b = (width - int(w_b)) // 2
y_offset_b = (height - int(h_b)) // 2

def FFT(im):
    # Step 1: Pad the red channel to the scaled size
    padded_red = np.zeros((h_r, w_r))
    padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]
    # Step 2: Crop the blue channel to the scaled size
    cropped_blue=np.zeros((h_b, w_b))
    cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]
    
    padded_B = np.zeros((int(h_r), int(w_r)))
    padded_B[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 2]
    # Step 3: Resize both channels back to the original dimensions
    scaled_red = cv2.resize(padded_red, (width, height), interpolation=cv2.INTER_LINEAR)
    scaled_blue = cv2.resize(cropped_blue, (width, height), interpolation=cv2.INTER_LINEAR)
    #<<----------------------End of rescaling preparation-------------------------------->>
    power1=1 # Power control
    # Power controlled intensity for different channels. For the simulation, we only need to change the rescaled red and blue channels to unrescaled channels.
    #R
    im_shift_r=fftshift(scaled_red**power1)
    #G
    im_shift_g=fftshift(im[:,:,1]**power1)
    #B
    im_shift_b=fftshift(scaled_blue**power1)
    
    # Multiply noise with the field of the preprocessed orignal objects.
    im_r_rand=exp_rand*np.sqrt(im_shift_r)
    im_g_rand=exp_rand*np.sqrt(im_shift_g)
    im_b_rand=exp_rand*np.sqrt(im_shift_b)
    # Forward FFT of different channels
    current_field_r =fftshift(fft2(im_r_rand ))
    current_field_g =fftshift(fft2(im_g_rand))
    current_field_b =fftshift(fft2(im_b_rand))
    
    iterations1=1 # Iteration number for the GS without window.
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
    
    # Phase modulation depth for experiment, and they are all 2 for simulation.
    rdp,gdp,bdp=1.78,2.56,3.5
    # Final optimized phase for encoding on SLM, turn angles to grayscales.
    optimized_phase_r = np.angle(current_field_r)
    phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/rdp)
    
    optimized_phase_g = np.angle(current_field_g)
    phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/gdp)
    
    optimized_phase_b = np.angle(current_field_b)
    phase_br_modi=(optimized_phase_b/np.pi+1)*(255/bdp)
    """Binarize"""
    gr,gg,gb=184,136,112
    optimized_phase_r_bi = np.where(optimized_phase_r < 0, 0, 1)
    phase_rr_modi_bi=optimized_phase_r_bi*gr
    # phase_rr_modi_mod=np.mod(phase_rr_modi,255)

    optimized_phase_g_bi = np.where(optimized_phase_g < 0, 0, 1)
    phase_gr_modi_bi=optimized_phase_g_bi*gg
    # phase_gr_modi_mod=np.mod(phase_gr_modi,255)

    optimized_phase_b_bi = np.where(optimized_phase_b < 0, 0, 1)
    phase_br_modi_bi=optimized_phase_b_bi*gb
    # phase_br_modi_mod=np.mod(phase_br_modi,255)
    """Lens"""
    lambda_r = 0.662e-6  
    lambda_g = 0.518e-6  
    lambda_b = 0.449e-6  
    arr_r=np.zeros((height, width))
    arr_g=np.zeros((height, width))
    arr_b=np.zeros((height, width))
    pixel_size=4.5e-6
    f=-2 # focal length (meters).
    
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
    
    im_modify_noL_b = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
    im_modify_noL_b[:,:,0] = phase_rr_modi_bi
    im_modify_noL_b[:,:,1] = phase_gr_modi_bi
    im_modify_noL_b[:,:,2] = phase_br_modi_bi
    
    im_modify_c_b = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
    im_modify_c_b[:,:,0] = phase_rr_modi_bi+arr_r_modified
    im_modify_c_b[:,:,1] = phase_gr_modi_bi+arr_g_modified
    im_modify_c_b[:,:,2] = phase_br_modi_bi+arr_b_modified
    
    return im_modify_noL, im_modify_c,im_modify_noL_b,im_modify_c_b

def crop(im_modify,name):
    y_offset=center_h-1080//2
    im_cropped=im_modify[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{name} It1_1_It2_0.png")
im1=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/L_V.png")
imnl,imc,imnl_b,imc_b=FFT(im1)
C=crop(imc, "F_V_L")
C_noL=crop(imnl, "F_V_nL")
# C_noL=crop(imnl[:,:,0], "F_V r nL")
# C_noL=crop(imnl[:,:,1], "F_V g nL")
# C_noL=crop(imnl[:,:,2], "F_V b nL")

im2=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/L_H0.png")
imnl,imc,imnl_b,imc_b=FFT(im2)
C=crop(imc, "F_H0_L")
C_noL=crop(imnl, "F_H0_nL")
C_noL=crop(imnl[:,:,0], "F_H0 r nL")
# C_noL=crop(imnl[:,:,1], "F_H0 g nL")
# C_noL=crop(imnl[:,:,2], "F_H0 b nL")

im3=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/L_H180.png")
imnl,imc,imnl_b,imc_b=FFT(im3)
C=crop(imc, "F_H180_L")
C_noL=crop(imnl, "F_H180_nL")
C_noL=crop(imnl[:,:,0], "F_H180 r nL")
# C_noL=crop(imnl[:,:,1], "F_H180 g nL")
# C_noL=crop(imnl[:,:,2], "F_H180 b nL")

end_t=time.time()
print(f"Time consuming {end_t-start_t}s")