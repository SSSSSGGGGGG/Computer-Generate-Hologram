# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os

height, width=1080,1920
arr_r=np.zeros((height, width),dtype="complex")
pixel_size=4.5e-6#4.5e-6
f=1 # meters
# Define wavelengths (in meters, for example)
lambda_r = 0.633e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.450e-6

center_h=height//2
center_w=width//2
max_exp_arg = 100  # Prevents overflow since exp(709) is close to the limit for float64

for i in range(height):
    for j in range(width):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        # arr_r[i, j] = np.pi * r**2 / (f * lambda_r)
        exp_arg = np.pi * r**2 / (f * lambda_r)
        arr_r[i, j] = np.exp(1j*exp_arg)  # Cap the argument min(exp_arg, max_exp_arg)
        
arr_r_fft=fftshift(fft2(arr_r))
I_lens=np.abs(arr_r_fft)**2
# Display results
plt.figure()
plt.imshow(I_lens,cmap="hot")
plt.colorbar()#label='Phase (radians)'
plt.axis('off')
plt.title("FFT of Lens for red laser")
plt.show()

# # Display results
# plt.figure()
# plt.imshow(exp_arg,cmap="gray")
# plt.colorbar()#label='Phase (radians)'
# plt.axis('off')
# plt.title("Lens for red laser im")
# plt.show()
# print(center_h,center_w)