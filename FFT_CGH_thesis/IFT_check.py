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

# File paths
original_path="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_z.png"
file_path1 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_z_GS_10_nL.png"
file_path2 = "C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/flowers_z_GS_(10, 'nL')_input_2_p2_2.png"

# Automatically retrieve image names
holo1_name = os.path.basename(file_path1)
holo2_name = os.path.basename(file_path2)

# Load holograms
y_offset=1920//2-1080//2
# im_cropped=im_modify[y_offset:y_offset+1080,:]
original=plt.imread(original_path)[y_offset:y_offset+1080,:][:,:,0]
holo1 = plt.imread(file_path1)
holo2 = plt.imread(file_path2)

# holo1_R=holo1[:,:,0]*255
# holo2_R=holo2[:,:,0]*255

# plt.figure()
# plt.title(f"{holo1_name}")
# plt.imshow(holo1_R, cmap="Reds")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.title(f"{holo2_name}")
# plt.imshow(holo2_R, cmap="Reds")
# plt.colorbar()
# plt.show()

holo1_R_angle=holo1[:,:,0]*1.8*np.pi
holo2_R_angle=holo2[:,:,0]*1.8*np.pi

Reconstruction_holo1 = fftshift(ifft2(np.exp(1j * holo1_R_angle)))
Reconstruction_holo2 = fftshift(ifft2(np.exp(1j * holo2_R_angle)))
I1=np.abs(Reconstruction_holo1)#♥**2
I1_normalized = (I1 - np.min(I1)) / (np.max(I1) - np.min(I1))
I2=np.abs(Reconstruction_holo2)#**2

plt.figure()
plt.title(f"{holo1_name}")
plt.imshow(I1,cmap="pink")
plt.colorbar()
# cbar =plt.colorbar()
# cbar.set_ticks([0, 1])
plt.axis("off")
plt.show()

plt.figure()
plt.title(f"{holo2_name}")

plt.imshow(I2,cmap="pink")
plt.axis("off")
plt.colorbar()
# cbar =plt.colorbar()
# cbar.set_ticks([0, 1])
plt.show()

# rgb_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
# rgb_image[:, :, 0] = (abs(Reconstruction_holo2)**2)*2550
# arr_r_im=Image.fromarray(rgb_image)
# arr_r_im.show()
# arr_r_im.save(f"Lens_rgb{f,height}.png")