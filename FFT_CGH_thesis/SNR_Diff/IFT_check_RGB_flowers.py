# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:55:01 2024

@author: gaosh
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os
import cv2
from scipy.interpolate import interp1d
from skimage import  color
# File paths
# original_path1="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/lemon_in.png"
original_path2="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960.png"

file_path1 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960_GS_25_1,nl.png"
file_path2 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960_GS_1_1,nl.png"

# Automatically retrieve image names
holo1_name = os.path.basename(file_path1)
holo2_name = os.path.basename(file_path2)

holo1 = plt.imread(file_path1)
holo2 = plt.imread(file_path2)


holo1_R_angle=holo1[:,:,0]*2*np.pi
holo2_R_angle=holo2[:,:,0]*2*np.pi

holo1_G_angle=holo1[:,:,1]*2*np.pi
holo2_G_angle=holo2[:,:,1]*2*np.pi

holo1_B_angle=holo1[:,:,2]*2*np.pi
holo2_B_angle=holo2[:,:,2]*2*np.pi

Reconstruction_holo1_r = fftshift(ifft2(np.exp(1j * holo1_R_angle)))
Reconstruction_holo2_r = fftshift(ifft2(np.exp(1j * holo2_R_angle)))

Reconstruction_holo1_g = fftshift(ifft2(np.exp(1j * holo1_G_angle)))
Reconstruction_holo2_g = fftshift(ifft2(np.exp(1j * holo2_G_angle)))

Reconstruction_holo1_b = fftshift(ifft2(np.exp(1j * holo1_B_angle)))
Reconstruction_holo2_b = fftshift(ifft2(np.exp(1j * holo2_B_angle)))

original2=plt.imread(original_path2)
original2_r=original2[:,:,0]/np.sum(original2[:,:,0])
original2_g=original2[:,:,1]/np.sum(original2[:,:,1])
original2_b=original2[:,:,2]/np.sum(original2[:,:,2])

factor=0.4#1#0.4#0.7#0.025
original2_nor=plt.imread(original_path2)
original2_nor_r=original2_r/np.max(original2_r*factor)
original2_nor_g=original2_g/np.max(original2_g*factor)
original2_nor_b=original2_b/np.max(original2_b*factor)



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

D1_r=np.sqrt(np.sum((original2_r-I1_r_normalized)**2))
D1_g=np.sqrt(np.sum((original2_g-I1_g_normalized)**2))
D1_b=np.sqrt(np.sum((original2_b-I1_b_normalized)**2))
D1=(D1_r+D1_g+D1_b)/3

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
# print(f"For I2 {np.sum(I1_r)},{np.sum(I1_g)},{np.sum(I1_b)}")
I2_rgb=np.zeros_like(holo1)
I2_rgb[:, :, 0] = I2_r_normalized/(np.max(I2_r_normalized)*factor)
I2_rgb[:, :, 1] = I2_g_normalized/(np.max(I2_g_normalized)*factor)
I2_rgb[:, :, 2] = I2_b_normalized/(np.max(I2_b_normalized)*factor)
# print(f"max R {np.max(I2_rgb[:, :, 0])}, max G {np.max(I2_rgb[:, :, 1])}, max B{np.max(I2_rgb[:, :, 2])}")
# print(f"max Rnorm {np.max(I2_r_normalized)}, max Gnorm {np.max(I2_g_normalized)}, max Bnorm{np.max(I2_b_normalized)}")
print(f"mean Rnorm {np.average(I2_r_normalized)}, mean Gnorm {np.average(I2_g_normalized)}, mean Bnorm{np.average(I2_b_normalized)}")
# print(f"max Rnorm {np.max(I2_r)}, max Gnorm {np.max(I2_g)}, max Bnorm{np.max(I2_b)}")
print(f"mean Rnorm {np.average(I2_r)}, mean Gnorm {np.average(I2_g)}, mean Bnorm{np.average(I2_b)}")

D2_r=np.sqrt(np.sum((original2_r-I2_r_normalized)**2))
D2_g=np.sqrt(np.sum((original2_g-I2_g_normalized)**2))
D2_b=np.sqrt(np.sum((original2_b-I2_b_normalized)**2))
D2=(D2_r+D2_g+D2_b)/3

print(f"D1 { D1}, D2 {D2}")

def hist(r,g,b,name):
    plt.figure()
    
    # Plot histograms for each channel
    plt.hist(r.flatten(), bins=30, color='red', alpha=0.5, label='Red')
    plt.hist(g.flatten(), bins=30, color='green', alpha=0.5, label='Green')
    plt.hist(b.flatten(), bins=30, color='blue', alpha=0.5, label='Blue')
    
    # Add mean intensity markers
    plt.axvline(np.average(r), color='red', linestyle='dashed', linewidth=2, label=f'Mean R')
    plt.axvline(np.average(g), color='green', linestyle='dashed', linewidth=2, label=f'Mean G')
    plt.axvline(np.average(b), color='blue', linestyle='dashed', linewidth=2, label=f'Mean B')
    
    # Add labels, legend, and title
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f"{name} RGB Histogram")
    plt.legend()
    
    # Save the histogram plot
    # plt.savefig(f"{name}_histogram.png")  # Correct function to save the plot
    
    # Show the plot
    plt.show()
# I1_hist=hist(I1_r_normalized,I1_g_normalized,I1_b_normalized,f"{holo1_name}")
# I2_hist=hist(I2_r_normalized,I2_g_normalized,I2_b_normalized,f"{holo2_name}")
# I2_hist=hist(original2_r,original2_g,original2_b,f"{holo2_name}")
# I2_r_hist=hist(original2_nor_r,original2_nor_g,original2_nor_b,f"{holo2_name} I2_r")



# plt.figure()
# plt.title(f"{holo1_name} no nor")
# plt.imshow(np.clip(original2_nor, 0, 1))
# plt.axis("off")
# plt.show()
# # plt.imsave(f"{holo1_name}_no nor f {factor}.png", np.clip(I2_rgb_o, 0, 1))

plt.figure()
plt.title(f"{holo1_name}")
plt.imshow(np.clip(I1_rgb, 0, 1)) #np.clip(I1_rgb, 0, 1)
plt.axis("off")
plt.show()
# plt.imsave(f"{holo1_name} f {factor}_win.png", np.clip(I1_rgb, 0, 1))
# I1_fin_hist=hist(np.clip(I1_rgb, 0, 1)[:,:,0],np.clip(I1_rgb, 0, 1)[:,:,1],np.clip(I1_rgb, 0, 1)[:,:,2],f"{holo1_name} f {factor}")


plt.figure()
plt.title(f"{holo2_name}")
plt.imshow(np.clip(I2_rgb, 0, 1))
plt.axis("off")
plt.show()
# plt.imsave(f"{holo2_name} f {factor}_win.png", np.clip(I2_rgb, 0, 1))
# I2_fin_hist=hist(np.clip(I2_rgb, 0, 1)[:,:,0],np.clip(I2_rgb, 0, 1)[:,:,1],np.clip(I2_rgb, 0, 1)[:,:,2],f"{holo2_name} f {factor}")

# plt.figure()
# plt.title(f"{holo2_name} rgb")
# plt.imshow(np.clip(I1_rgb, 0, 1)[:,:,0]-np.mod(I1_rgb, 1)[:,:,0])  #np.clip(I2_rgb, 0, 1)
# plt.axis("off")
# plt.colorbar()
# plt.show()



# plt.figure()
# plt.title(f"{holo2_name} B")
# plt.imshow(I1_b_normalized)
# plt.axis("off")
# plt.show()

# # # # Define custom curve control points
# # # x = [0.0, 0.001,0.03,0.005,0.01, 0.016,0.7,0.9, 1.0]  # Input intensities (original image)
# # # y = [0.2, 0.25,0.3,0.3,0.5,0.65, 0.8, 0.9,1.0]  # Output intensities (desired mapping)
# # # curve = interp1d(x, y, kind="cubic", fill_value="extrapolate")
# # # adjusted_np = np.clip(curve(I2_normalized), 0, 1)
# # # # Create the interpolation function (cubic for smooth transitions)
# # # plt.figure()
# # # plt.imshow(adjusted_np,cmap="pink")
# # # plt.axis("off")
# # # plt.colorbar()
# # # plt.show()