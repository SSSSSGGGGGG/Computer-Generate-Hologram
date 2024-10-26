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
# file_n = "R"   
p_X = 92#81#79#128
max_phase = 2 * np.pi
p_Y = 1080#640#400#540
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
fft_grating_X = fftshift(fft2(blazed_grating_X))
I_X = np.abs(fft_grating_X)**2
I_X = I_X / np.sum(I_X)
# Fourier Transform and Visualization

x = np.linspace(-0.5*width, 0.5*width, width)
y = np.linspace(-0.5*height, 0.5*height, height)
X, Y = np.meshgrid(x, y)
beam_width = 150
Guassia_amplitude = np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * beam_width**2))
Guassia_amplitude=Guassia_amplitude/np.max(Guassia_amplitude)
fft_grating_X_G = fftshift(fft2(blazed_grating_X*Guassia_amplitude))
I_X_G = np.abs(fft_grating_X_G)**2
I_X_G = I_X_G / np.sum(I_X_G)
print(f"fx1={delta_X}")
print(f"position for 1st and 2nd {int(width/2+delta_X)-960,int(width/2+2*delta_X)-960}")
print(f"maximum={np.max(I_X)},0th_X={I_X[int(height/2),int(width/2)]}")
print(f"1th_X={I_X[int(height/2),int(width/2+delta_X)]},2nd_X={I_X[int(height/2),int(width/2+2*delta_X)]}")

fft_grating_Y = fftshift(fft2(blazed_grating_Y))
I_Y = np.abs(fft_grating_Y)**2
I_Y = I_Y / np.sum(I_Y)

fft_grating_Y_G = fftshift(fft2(blazed_grating_Y*Guassia_amplitude))
I_Y_G = np.abs(fft_grating_Y_G)**2
I_Y_G = I_Y_G / np.sum(I_Y_G)
print(f"fy1={delta_Y}")
print(f"position for 1st and 2nd {int(height/2+delta_Y)-540,int(height/2+2*delta_Y)-540}")
print(f"maximum={np.max(I_Y)},0th_X={I_Y[int(height/2),int(width/2)]}")
print(f"1th_Y={I_Y[int(height/2+delta_Y+1),int(width/2)]},2nd_Y={I_Y[int(height/2+2*delta_Y+1),int(width/2)]}")

fft_grating = fftshift(fft2(blazed_grating_Y*blazed_grating_X))
I = np.abs(fft_grating)**2
I = I / np.sum(I)

fft_grating_G = fftshift(fft2(blazed_grating_Y*blazed_grating_X*Guassia_amplitude))
I_G = np.abs(fft_grating_G)**2
I_G = I_G / np.sum(I_G)
# print(f"position for 1st and 2nd {int(height/2+delta_Y)-540,int(height/2+2*delta_Y)-540}")
print(f"maximum={np.max(I)},0th_X={I[int(height/2),int(width/2)]}")
print(f"1th={I[int(height/2+delta_Y+1),int(width/2++delta_X)]},2nd={I[int(height/2+2*delta_Y+1),int(width/2+2*delta_X)]}")

# Cropped FFT Magnitude
center_h, center_w = height // 2, width // 2
crop_size = 200
start_h, end_h = center_h - crop_size // 2, center_h + crop_size // 2
start_w, end_w = center_w - crop_size // 4, center_w + crop_size // 4
fft_magnitude_cropped_X = I_X_G[start_h:end_h, start_w:end_w]
fft_magnitude_cropped_Y = I_Y_G[start_h:end_h, start_w:end_w]
fft_magnitude_cropped = I_G[start_h:end_h, start_w:end_w]
I_cropped=I_X[int(start_h):int(end_h), int(start_w):int(end_w)]
# Plot Results
# plt.figure()
# plt.imshow(I_X_G, cmap='hot')
# plt.title('Blazed Phase')
# plt.colorbar()
# plt.show()
plt.figure()
plt.imshow(I_Y_G, cmap='hot')
plt.title('Blazed Phase')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(I_cropped, cmap='hot')
plt.colorbar()
plt.show()

# plt.imsave(f"Blazed_X p={p_X,max_phase/np.pi}pi.png", blazed_grating_X.real, cmap='gray')
# plt.imsave(f"Blazed_Y p={p_Y,max_phase/np.pi}pi.png", blazed_grating_Y.real, cmap='gray')
# plt.imsave(f"FFT of blazed p={p_X,max_phase/np.pi}pi.png", fft_magnitude_cropped, cmap='hot')
# plt.imsave(f"FFT of blazed_Y p={p_Y,max_phase/np.pi}pi.png", fft_magnitude_cropped_Y, cmap='hot')
plt.imsave(f"FFT of blazeds pV_{p_Y}px pH_{p_X}px {max_phase/np.pi}pi.png", fft_magnitude_cropped, cmap='hot')