# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

width, height = 1920, 1080    
file_n = "R"   
p_X = 92
max_phase = 2 * np.pi
p_Y = 400
delta_X = 1 / p_X * width
delta_Y = 1 / p_Y * height
blazed_phase_X=np.ones((height, width))
blazed_phase_Y=np.ones((height, width))
blazed_grating_X = np.ones((height, width), dtype="complex128")
blazed_grating_Y = np.ones((height, width), dtype="complex128")

stripe_width = p_X  # period
stripe_width_v = p_Y  # vertical period

# Horizontal grating pattern
loop = width // stripe_width
reminder = width % stripe_width
interval = max_phase / (stripe_width - 1)

for x1 in range(height):
    for i1 in range(stripe_width):
        for k in range(int(loop)):
            idx = i1 + k * stripe_width
            if idx < width:  # Ensure we don't exceed width
                blazed_phase_X[x1, idx] = i1 * interval/np.pi
                blazed_grating_X[x1, idx] = np.exp(1j * (i1 * interval))
    # Handle remaining columns if width is not divisible by stripe_width
    for i2 in range(reminder):
        idx = i2 + (loop * stripe_width)
        if idx < width:
            blazed_phase_X[x1, idx] = i2 * interval/np.pi
            blazed_grating_X[x1, idx] = np.exp(1j * (i2 * interval))

# Vertical grating pattern
loop_v = height // stripe_width_v
reminder_v = height % stripe_width_v
interval_v = max_phase / (stripe_width_v - 1)

for x1 in range(width):
    for i1 in range(stripe_width_v):
        for k in range(int(loop_v)):
            idx = i1 + k * stripe_width_v
            if idx < height:  # Ensure we don't exceed height
                blazed_phase_Y[idx, x1] = i1 * interval_v/np.pi
                blazed_grating_Y[idx, x1] = np.exp(1j * (i1 * interval_v))
    # Handle remaining rows if height is not divisible by stripe_width_v
    for i2 in range(reminder_v):
        idx = i2 + (loop_v * stripe_width_v)
        if idx < height:
            blazed_phase_Y[idx, x1] = i2 * interval_v/np.pi
            blazed_grating_Y[idx, x1] = np.exp(1j * (i2 * interval_v))

# Fourier Transform and Visualization
fft_grating_X_G = fftshift(fft2(blazed_grating_X))
I_X_G = np.abs(fft_grating_X_G)**2
I_X_G = I_X_G / np.sum(I_X_G)

fft_grating_X = fftshift(fft2(blazed_grating_X))
I_X = np.abs(fft_grating_X)**2
I_X = I_X / np.sum(I_X)
print(f"fx1={delta_X}")
print(f"position for 1st and 2nd {int(width/2+delta_X)-960,int(width/2+2*delta_X)-960}")
print(f"maximum={np.max(I_X)},0th_X={I_X[int(height/2),int(width/2)]}")
print(f"1th_X={I_X[int(height/2-1),int(width/2+delta_X)]},2nd_X={I_X[int(height/2),int(width/2+2*delta_X)]}")

# Cropped FFT Magnitude
center_h, center_w = height // 2, width // 2
crop_size = 200
start_h, end_h = center_h - crop_size // 2, center_h + crop_size // 2
start_w, end_w = center_w - crop_size // 2, center_w + crop_size // 2
fft_magnitude_cropped = I_X_G[start_h:end_h, start_w:end_w]

# Plot Results
plt.figure()
plt.imshow(I_X_G, cmap='hot')
plt.title('Blazed Phase')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fft_magnitude_cropped, cmap='hot')
plt.colorbar()
plt.show()

plt.imsave(f"FFT of blazed p={p_X,max_phase}pi.png", fft_magnitude_cropped, cmap='hot')
