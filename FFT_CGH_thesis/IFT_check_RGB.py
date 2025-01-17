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

file_path1 = "C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/holograms/flowers_960_wh_(10, 'nL')_p3_1_input1.png"
file_path2 = "C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/holograms/flowers_960_GS_10_nL.png"

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

I1_r=np.abs(Reconstruction_holo1_r)#**2
I1_r_normalized = I1_r / np.sum(I1_r)

I1_g=np.abs(Reconstruction_holo1_g)#**2
I1_g_normalized = I1_g / np.sum(I1_g)

I1_b=np.abs(Reconstruction_holo1_b)#**2
I1_b_normalized = I1_b / np.sum(I1_b)

I1_rgb=np.zeros_like(holo1)
I1_rgb[:, :, 0] = I1_r_normalized
I1_rgb[:, :, 1] = I1_g_normalized
I1_rgb[:, :, 2] = I1_b_normalized

original2=plt.imread(original_path2)
original2_r=original2[:,:,0]/np.sum(original2[:,:,0])
original2_g=original2[:,:,1]/np.sum(original2[:,:,1])
original2_b=original2[:,:,2]/np.sum(original2[:,:,2])

D1_r=np.sqrt(np.sum((original2_r-I1_r_normalized)**2))
D1_g=np.sqrt(np.sum((original2_g-I1_g_normalized)**2))
D1_b=np.sqrt(np.sum((original2_b-I1_b_normalized)**2))
D1=(D1_r+D1_g+D1_b)/3


# Convert to image
# I1_rgb_im = Image.fromarray((I1_rgb*255).astype(np.uint8), 'RGB')

# # Save or display the image
# I1_rgb_im.show()  # Display the image


I2_r=np.abs(Reconstruction_holo2_r)#**2
I2_r_normalized = I2_r / np.sum(I2_r)
I2_g=np.abs(Reconstruction_holo2_g)#**2
I2_g_normalized = I2_g / np.sum(I2_g)
I2_b=np.abs(Reconstruction_holo2_b)#**2
I2_b_normalized = I2_b / np.sum(I2_b)

I2_rgb=np.zeros_like(holo1)
I2_rgb[:, :, 0] = I2_r_normalized
I2_rgb[:, :, 1] = I2_g_normalized
I2_rgb[:, :, 2] = I2_b_normalized

D2_r=np.sqrt(np.sum((original2_r-I2_r_normalized)**2))
D2_g=np.sqrt(np.sum((original2_g-I2_g_normalized)**2))
D2_b=np.sqrt(np.sum((original2_b-I2_b_normalized)**2))
D2=(D2_r+D2_g+D2_b)/3

print(f"D1 { D1}, D2 {D2}")

# plt.figure()
# plt.title(f"{holo1_name} original")
# plt.imshow(original2)
# plt.axis("off")
# plt.show()


plt.figure()
plt.title(f"{holo1_name} rgb")
plt.imshow(I1_rgb/np.max(I1_rgb))
plt.axis("off")
plt.show()
# plt.imsave(f"{holo1_name}_rgb.png", I1_rgb/np.max(I1_rgb))


plt.figure()
plt.title(f"{holo2_name} rgb")
plt.imshow(I2_rgb/np.max(I2_rgb))
plt.axis("off")
plt.show()
# plt.imsave(f"{holo2_name}_rgb.png", I2_rgb/np.max(I2_rgb))

# plt.figure()
# plt.title(f"{holo2_name} G")
# plt.imshow(I1_g_normalized)
# plt.axis("off")
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