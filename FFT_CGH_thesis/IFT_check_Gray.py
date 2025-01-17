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
original_path1="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/lemon_in.png"
original_path2="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960.png"

file_path1 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960_G_GS_(10, 'nL')_input_2_p2_2.png"
file_path2 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960_G_GS_(25, 'nL')_input_2_p2_2.png"
file_path3 ="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_960_G_GS_(50, 'nL')_input_2_p2_2.png"
# Automatically retrieve image names
holo1_name = os.path.basename(file_path1)
holo2_name = os.path.basename(file_path2)
holo3_name = os.path.basename(file_path3)


original2=plt.imread(original_path2)[...,:3]
im2 = color.rgb2gray(original2)
im2_normalized=im2 / np.sum(im2)
im2_normalized_1=im2_normalized/np.max(im2_normalized)
w,h=im2.shape

holo1 = plt.imread(file_path1)
holo2 = plt.imread(file_path2)
holo3 = plt.imread(file_path3)

holo1_R_angle=holo1[:,:,0]*2*np.pi
holo2_R_angle=holo2[:,:,0]*2*np.pi
holo3_R_angle=holo3[:,:,0]*2*np.pi

Reconstruction_holo1 = fftshift(ifft2(np.exp(1j * holo1_R_angle)))
Reconstruction_holo2 = fftshift(ifft2(np.exp(1j * holo2_R_angle)))
Reconstruction_holo3 = fftshift(ifft2(np.exp(1j * holo3_R_angle)))

I1=np.abs(Reconstruction_holo1)#**2
I1_normalized=I1 / np.sum(I1)
I1_normalized_1 = I1_normalized/np.max(I1_normalized)
D1=np.sqrt(np.sum((im2_normalized_1-I1_normalized)**2))/(h*w)

I2=np.abs(Reconstruction_holo2)#**2
I2_normalized=I2 / np.sum(I2)
I2_normalized_1 = I2_normalized/np.max(I2_normalized)
D2=np.sqrt(np.sum((im2_normalized_1-I2_normalized)**2))/(h*w)

I3=np.abs(Reconstruction_holo3)#**2
I3_normalized=I3 / np.sum(I3)
I3_normalized_1 = I3_normalized/np.max(I3_normalized)
D3=np.sqrt(np.sum((im2_normalized_1-I3_normalized)**2))/(h*w)

print(f"D1 { D1}, D2 {D2},D3 {D3}")


plt.figure()
plt.title(f"original")
plt.imshow(im2_normalized,vmin=0, cmap="rainbow")
plt.colorbar()
plt.show()

plt.figure()
plt.title(f"{holo1_name}")
plt.imshow(I1_normalized,vmin=0,cmap="rainbow")
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.title(f"{holo2_name}")
plt.imshow(I2_normalized,vmin=0,cmap="rainbow")
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure()
plt.title(f"{holo3_name}")
plt.imshow(I3_normalized,vmin=0,cmap="rainbow")
plt.axis("off")
plt.colorbar()
plt.show()
# # Define custom curve control points
# x = [0.0, 0.001,0.03,0.005,0.01, 0.016,0.7,0.9, 1.0]  # Input intensities (original image)
# y = [0.2, 0.25,0.3,0.3,0.5,0.65, 0.8, 0.9,1.0]  # Output intensities (desired mapping)
# curve = interp1d(x, y, kind="cubic", fill_value="extrapolate")
# adjusted_np = np.clip(curve(I2_normalized), 0, 1)
# # Create the interpolation function (cubic for smooth transitions)
# plt.figure()
# plt.imshow(adjusted_np,cmap="pink")
# plt.axis("off")
# plt.colorbar()
# plt.show()