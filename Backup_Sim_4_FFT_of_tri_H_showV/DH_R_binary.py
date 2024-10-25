# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

phase = 0.64 * np.pi   
G1 = np.exp(1j * phase)
G2 = np.exp(1j * 0) 
file_n = "R_tri"   
width, height = 1920, 1080
binary_phase = np.ones((height, width),dtype="complex128")
stripe_width = 64  # period
spacing = stripe_width * 2   # 2 times period
delta_bi=0.5*height/stripe_width

# Determine loop and remainder for height
loop_h = height // stripe_width
reminder_h = height % stripe_width

# Vertical diffraction pattern distribution
if reminder_h == 0:
    for x in range(width):
        for i in range(0, height, spacing):
            for j in range(stripe_width):
                if i + j < height:
                    binary_phase[i + j, x] = G1
else:
    for x in range(width):
        for i in range(0, height, spacing):
            for j in range(stripe_width):
                if i + j < height:
                    binary_phase[i + j, x] = G1
        # Handle remaining pixels for the reminder height
        i2 = 0
        while i2 < reminder_h - 1:
            if (i2 + 1 + (loop_h) * stripe_width) < height:
                binary_phase[i2 + 1 + (loop_h) * stripe_width, x] = G1
            i2 += 1
fft_grating_binary = fftshift(fft2(binary_phase))#*Guassia_amplitude
I_binary=np.abs(fft_grating_binary)**2
sumI=np.sum(I_binary)
I_binary_nor=I_binary/sumI

I_0=I_binary_nor[540,960]
I_1=I_binary_nor[540+int(delta_bi),960]
I_n1=I_binary_nor[540-int(delta_bi),960]
print(f"f1={delta_bi}, I_1={540+int(delta_bi)},I_1={540-int(delta_bi)}")
print(f"I_0={I_0},I_1={I_1},I_-1={I_n1} ")
# Display the result
plt.figure()
plt.imshow(binary_phase.real, cmap="gray")
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(I_binary_nor, cmap='hot')  #,vmin=0, vmax=0.2
plt.title('FFT of the binary phase grating * Beam')
plt.colorbar()
