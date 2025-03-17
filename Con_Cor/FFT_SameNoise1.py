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
os.chdir("C:/Users/Laboratorio/MakeHologram/Con_Cor")
filename="Lotus" 
im_o=plt.imread(f"{filename}.png")
h,w=im_o[:,:,0].shape

height=1920
width=1920

center_h=height//2
center_w=width//2
# Genaration of random phase noise.
rand=np.random.uniform(0, 1, (h, w,3))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)

im_noise=im_o[:,:,:3]*exp_rand
im_c = np.zeros((height, width, 3), dtype=complex)
im_c[center_h-150:center_h+150,center_w-150:center_w+150]=im_noise

length=-400
im_v = np.zeros((height, width, 3), dtype=complex)
im_v[center_h-150+length:center_h+150+length,center_w-150:center_w+150]=im_noise

im_l = np.zeros((height, width, 3), dtype=complex)
im_l[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise

im_noise_flipped = np.flip(im_noise, axis=(0, 1))  # Flip both vertically & horizontally
im_l_fl = np.zeros((height, width, 3), dtype=complex)
im_l_fl[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise_flipped

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

x_offset_r = (int(w_r)-width) // 2
y_offset_r = (int(h_r)-height) // 2

x_offset_b = (width - int(w_b)) // 2
y_offset_b = (height - int(h_b)) // 2

def rescale(im):
    # Step 1: Pad the red channel to the scaled size
    padded_red = np.zeros((h_r, w_r), dtype=complex)
    padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]
    # Step 2: Crop the blue channel to the scaled size
    cropped_blue=np.zeros((h_b, w_b), dtype=complex)
    cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]
    # Step 3: Resize both channels back to the original dimensions
    real_partr = cv2.resize(np.real(padded_red), (width, height), interpolation=cv2.INTER_LINEAR)
    imag_partr = cv2.resize(np.imag(padded_red), (width, height), interpolation=cv2.INTER_LINEAR)
    scaled_red = real_partr + 1j * imag_partr
    
    real_partb = cv2.resize(np.real(cropped_blue), (width, height), interpolation=cv2.INTER_LINEAR)
    imag_partb = cv2.resize(np.imag(cropped_blue), (width, height), interpolation=cv2.INTER_LINEAR)
    scaled_blue = real_partb + 1j * imag_partb  # Recombine into complex format
    #R
    im_shift_r=fftshift(scaled_red)
    #G
    im_shift_g=fftshift(im[:,:,1])
    #B
    im_shift_b=fftshift(scaled_blue)
    
    return im_shift_r,im_shift_g,im_shift_b

rdp,gdp,bdp=1.85,2.63,3.55
def FFT(im_shift_r,im_shift_g,im_shift_b):    
    im_r_rand=np.sqrt(im_shift_r)
    im_g_rand=np.sqrt(im_shift_g)
    im_b_rand=np.sqrt(im_shift_b)
    # FFT of different channels
    current_field_r =fftshift(fft2(im_r_rand ))
    current_field_g =fftshift(fft2(im_g_rand))
    current_field_b =fftshift(fft2(im_b_rand))  
    # Phase modulation depth for experiment.
    
    # Convert phase to grayscale.
    optimized_phase_r = np.angle(current_field_r)
    phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/rdp)
    
    optimized_phase_g = np.angle(current_field_g)
    phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/gdp)
    
    optimized_phase_b = np.angle(current_field_b)
    phase_br_modi=(optimized_phase_b/np.pi+1)*(255/bdp)
    
    """Binarize"""
    type_of="tri"
    gr,gg,gb=185,185,185
    optimized_phase_r_bi = np.where(optimized_phase_r < 0, 0, 1)
    phase_rr_modi_bi=optimized_phase_r_bi*gr
       
    optimized_phase_g_bi = np.where(optimized_phase_g < 0, 0, 1)
    phase_gr_modi_bi=optimized_phase_g_bi*gg
    
    optimized_phase_b_bi = np.where(optimized_phase_b < 0, 0, 1)
    phase_br_modi_bi=optimized_phase_b_bi*gb
    
    return phase_rr_modi,phase_gr_modi,phase_br_modi,phase_rr_modi_bi,phase_gr_modi_bi,phase_br_modi_bi

"""Lens"""
lambda_r = 0.662e-6  
lambda_g = 0.518e-6  
lambda_b = 0.449e-6  
arr_r=np.zeros((height, width))
arr_g=np.zeros((height, width))
arr_b=np.zeros((height, width))
pixel_size=4.5e-6
f=-2 # focal length (meters).

# Calculate the phase of lenses.
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
# # Save and crop the corresponding RGB CGHs with/without lens for experiment and simulation, respectively.
# im_modify_noL = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_noL[:,:,0] = phase_rr_modi
# im_modify_noL[:,:,1] = phase_gr_modi
# im_modify_noL[:,:,2] = phase_br_modi

# im_modify_c = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_c[:,:,0] = phase_rr_modi+arr_r_modified
# im_modify_c[:,:,1] = phase_gr_modi+arr_g_modified
# im_modify_c[:,:,2] = phase_br_modi+arr_b_modified

# im_modify_c_r = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_c_r[:,:,0] = phase_rr_modi+arr_r_modified
# im_modify_c_g = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_c_g[:,:,1] = phase_gr_modi+arr_g_modified
# im_modify_c_b = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_c_b[:,:,2] = phase_br_modi+arr_b_modified

# im_modify_noL_bi = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_noL_bi[:,:,0] = phase_rr_modi_bi
# im_modify_noL_bi[:,:,1] = phase_gr_modi_bi
# im_modify_noL_bi[:,:,2] = phase_br_modi_bi

# im_modify_c_bi = np.zeros((height, width, 3), dtype=np.float32)
# im_modify_c_bi[:,:,0] = phase_rr_modi_bi+arr_r_modified
# im_modify_c_bi[:,:,1] = phase_gr_modi_bi+arr_g_modified
# im_modify_c_bi[:,:,2] = phase_br_modi_bi+arr_b_modified

# return im_modify_noL, im_modify_c,im_modify_noL_bi,im_modify_c_bi,im_modify_c_r,im_modify_c_g,im_modify_c_b,type_of

# def crop(im_modify,name):
#     y_offset=center_h-1080//2
#     im_cropped=im_modify[y_offset:y_offset+1080,:]
#     im_cropped = im_cropped.astype(np.uint8)
#     im_modi = Image.fromarray(im_cropped)
#     im_modi.save(f"{name} n_1.png")
# imnl,imc,imnl_bi,imc_bi,imc_r,imc_g,imc_b,t=FFT(im_v)
# # C=crop(imc, f"F_V_L_({t})")
# # C_noL=crop(imnl, f"F_V_nL_({t})")
# C=crop(imc_bi, f"F_V_Bi({t})_L")
# C_noL=crop(imnl_bi, f"F_V_Bi({t})_nL")

# imnl,imc,imnl_bi,imc_bi,imc_r,imc_g,imc_b,t=FFT(im_l)
# # C=crop(imc, f"F_H0_L_({t})")
# # C_noL=crop(imnl, f"F_H0_nL_({t})")
# # C_noL=crop(imnl[:,:,0], f"F_H0 r nL_({t})")
# # C_noL=crop(imnl[:,:,1], f"F_H0 g nL_({t})")
# # C_noL=crop(imnl[:,:,2], f"F_H0 b nL_({t})")

# C=crop(imc_bi, f"F_H0__bi({t}) L")
# C_noL=crop(imnl_bi, f"F_H0_bi({t})_nL")

# imnl,imc,imnl_bi,imc_bi,imc_r,imc_g,imc_b,t=FFT(im_l_fl)
# # C=crop(imc, f"F_H180_L_({t})")
# # C_noL=crop(imnl, f"F_H180_nL_({t})")
# # C_noL=crop(imnl[:,:,0], f"F_H180 r nL_({t})")
# # C_noL=crop(imnl[:,:,1], f"F_H180 g nL_({t})")
# # C_noL=crop(imnl[:,:,2], f"F_H180 b nL_({t})")
# # C_noL=crop(imc_r, f"F_H180 r L_({t})")
# # C_noL=crop(imc_g, f"F_H180 g L_({t})")
# # C_noL=crop(imc_b, f"F_H180 b L_({t})")
# C=crop(imc_bi,f"F_H180_bi({t}) L")
# C_noL=crop(imnl_bi, f"F_H180_bi({t}) nL")

# end_t=time.time()
# print(f"Time consuming {end_t-start_t}s")