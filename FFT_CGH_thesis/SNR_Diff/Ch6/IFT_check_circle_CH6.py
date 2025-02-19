# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:55:01 2024

@author: gaosh
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift,ifft2
import os

# Read the path of original image.
original_path2="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6/One circle_1024.png"
# Read the path of RGB CGHs.
file_path1 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6/One circle_1024_GS_n1_30,n2_5_nl.png"
file_path2 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6/One circle_1024_GS_n1_35,n2_5_nl.png"
# Automatically retrieve image names
holo1_name = os.path.basename(file_path1)
holo2_name = os.path.basename(file_path2)
# Read RGB CGHs and original image.
holo1 = plt.imread(file_path1)
holo2 = plt.imread(file_path2)
original2=plt.imread(original_path2)
original2_r=original2[:,:,0]/np.sum(original2[:,:,0])
original2_g=original2[:,:,1]/np.sum(original2[:,:,1])
original2_b=original2[:,:,2]/np.sum(original2[:,:,2])

height=holo1.shape[0]
width=holo1.shape[1]
# Convert grayscales to angles.
holo1_R_angle=holo1[:,:,0]*2*np.pi
holo2_R_angle=holo2[:,:,0]*2*np.pi

holo1_G_angle=holo1[:,:,1]*2*np.pi
holo2_G_angle=holo2[:,:,1]*2*np.pi

holo1_B_angle=holo1[:,:,2]*2*np.pi
holo2_B_angle=holo2[:,:,2]*2*np.pi
# Compute inverse FFT2 of phase holograms to get reconstrutions.
Reconstruction_holo1_r = fftshift(ifft2(np.exp(1j * holo1_R_angle)))
Reconstruction_holo2_r = fftshift(ifft2(np.exp(1j * holo2_R_angle)))

Reconstruction_holo1_g = fftshift(ifft2(np.exp(1j * holo1_G_angle)))
Reconstruction_holo2_g = fftshift(ifft2(np.exp(1j * holo2_G_angle)))

Reconstruction_holo1_b = fftshift(ifft2(np.exp(1j * holo1_B_angle)))
Reconstruction_holo2_b = fftshift(ifft2(np.exp(1j * holo2_B_angle)))

l=560 
c_w,c_h=width//2,height//2
lh,lw=height-2*l,width-2*l # The size of the window.

# Calculate SNR and Diff for different RGB CGHs, and save the reconstrutions by defined factor, which is designed for visulizing noise.
def changef(f):
    factor=f
       
    I1_r=np.abs(Reconstruction_holo1_r)**2
    I1_r_normalized = I1_r / np.sum(I1_r)
    I1_g=np.abs(Reconstruction_holo1_g)**2
    I1_g_normalized = I1_g / np.sum(I1_g)
    I1_b=np.abs(Reconstruction_holo1_b)**2
    I1_b_normalized = I1_b / np.sum(I1_b)
    
    I1_rgb=np.zeros_like(holo1)
    I1_rgb[:, :, 0] = I1_r_normalized/(np.max(I1_r_normalized)*factor)
    I1_rgb[:, :, 1] = I1_g_normalized/(np.max(I1_g_normalized)*factor)
    I1_rgb[:, :, 2] = I1_b_normalized/(np.max(I1_b_normalized)*factor)
    #SNR in the window.
    D1_r_snr=np.sum(I1_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_r)-np.sum(I1_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    D1_g_snr=np.sum(I1_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_g)-np.sum(I1_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    D1_b_snr=np.sum(I1_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_b)-np.sum(I1_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    D1_snr=(D1_r_snr+D1_g_snr+D1_b_snr)/3
    
    diff_r1=original2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_r_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    diff_g1=original2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_g_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    diff_b1=original2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_b_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    # Diff in the window with respect to the original.
    D1_r=np.sqrt(np.sum((diff_r1)**2))
    D1_g=np.sqrt(np.sum((diff_g1)**2))
    D1_b=np.sqrt(np.sum((diff_b1)**2))
    D1=(D1_r+D1_g+D1_b)/3
    # The same calculations for the second RGB CGHs.
    print(f"For I2 ")
    I2_r=np.abs(Reconstruction_holo2_r)**2
    I2_r_normalized = I2_r / np.sum(I2_r)
    I2_g=np.abs(Reconstruction_holo2_g)**2
    I2_g_normalized = I2_g / np.sum(I2_g)
    I2_b=np.abs(Reconstruction_holo2_b)**2
    I2_b_normalized = I2_b / np.sum(I2_b)
    
    I2_rgb_o=np.zeros_like(holo1)
    I2_rgb_o[:, :, 0] = I2_r_normalized
    I2_rgb_o[:, :, 1] = I2_g_normalized
    I2_rgb_o[:, :, 2] = I2_b_normalized

    I2_rgb=np.zeros_like(holo1)
    I2_rgb[:, :, 0] = I2_r_normalized/(np.max(I2_r_normalized)*factor)
    I2_rgb[:, :, 1] = I2_g_normalized/(np.max(I2_g_normalized)*factor)
    I2_rgb[:, :, 2] = I2_b_normalized/(np.max(I2_b_normalized)*factor)
    #SNR
    D2_r_snr=np.sum(I2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I2_r)-np.sum(I2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    D2_g_snr=np.sum(I2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I2_g)-np.sum(I2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    D2_b_snr=np.sum(I2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I2_b)-np.sum(I2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    D2_snr=(D2_r_snr+D2_g_snr+D2_b_snr)/3
    
    diff_r2=original2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I2_r_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    diff_g2=original2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I2_g_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    diff_b2=original2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I2_b_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    # Diff
    D2_r=np.sqrt(np.sum((diff_r2)**2))
    D2_g=np.sqrt(np.sum((diff_g2)**2))
    D2_b=np.sqrt(np.sum((diff_b2)**2))
    D2=(D2_r+D2_g+D2_b)/3
    print(f"D1_snr { D1_snr}, D2_snr {D2_snr}")
    print(f"D1 { D1}, D2 {D2}")
    # Save the correspnding reconstrutions.
    plt.imsave(f"{holo1_name} f {factor}_win.png", np.clip(I1_rgb, 0, 1))
    plt.imsave(f"{holo2_name} f {factor}_win.png", np.clip(I2_rgb, 0, 1))
    
# Call the funtion "changef" by factor "0.05" and "1".
f2=changef(0.05)
f3=changef(1)
