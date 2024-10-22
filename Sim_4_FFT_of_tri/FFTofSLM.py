# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt


Rslits=np.ones((1080,1920))
height,width=Rslits.shape
center_h=int(height/2)
center_w=int(width/2)
period=128*4.6e-6
half_period=int(period/2)
phase=0.63*np.pi
# for i in range(0, width, 2*period):  
#     # print(i,i +period)
#     Rslits[0:height,i:i +period]=np.exp(1j*phase)  
# # plt.imsave(f"Vertical slits p_{period}.png",Rslits,cmap='Reds')

# Rslits_fft=fftshift(fft2(Rslits))
# Rslits_mag=abs(Rslits_fft)/np.max(abs(Rslits_fft))
# plt.imsave(f"FFT_Vslits p_{period}.png", Rslits_mag,cmap='hot')
# Create a spatial grid
x = np.linspace(0, width*4.6e-6, width)
y = np.linspace(0, height*4.6e-6, height)
X, Y = np.meshgrid(x, y)
beam_width = 200  # Width of the Gaussian beam

# Create a sinusoidal phase grating
phase_grating = np.exp(1j * phase * np.sin(2 * np.pi * X / period))

amplitude = np.exp(-((X/4.6e-6 - center_w)**2 + (Y/4.6e-6 - center_h)**2) / (2 * beam_width**2))
# Compute the 2D Fourier Transform and shift zero frequency component to the center
fft_grating = fftshift(fft2(amplitude*phase_grating))

# Compute the magnitude of the Fourier Transform
fft_magnitude = np.abs(fft_grating)
# Define the center region to crop
center_h = height // 2
center_w = width // 2
crop_size = 200  # Size of the cropped region (200x200 pixels)

# Crop the FFT magnitude
start_h = center_h - crop_size // 2
end_h = center_h + crop_size // 2
start_w = center_w - crop_size // 2
end_w = center_w + crop_size // 2
fft_magnitude_cropped = fft_magnitude[start_h:end_h, start_w:end_w]
# Normalize the magnitude for visualization
# fft_magnitude = fft_magnitude / np.max(fft_magnitude)
# square_h=np.zeros((512,512))
# width_h,height_h=square_h.shape
# center_h=int(width_h/2)
# square_h[center_h-width_c:center_h+width_c,center_h-width_c:center_h+width_c]=1
# for i in range(center_h-width_c, center_h+width_c, period):  
#     square_h[i:i + half_period,center_h - width_c:center_h+width_c] = 0  
# plt.imsave(f"Horizontal slits p_{period}.png", square_h,cmap='gray')        

# square_h_fft=fftshift(fft2(square_h))
# square_h_mag=abs(square_h_fft)/np.max(abs(square_h_fft))
# plt.imsave(f"FFT_Hslits p_{period}.png", square_h_mag,cmap='hot')

# SLM=np.zeros((512,512))
# width_s,height_s=SLM.shape
# center_s=int(width_s/2)
# SLM[center_s-width_c:center_s+width_c,center_s-width_c:center_s+width_c]=1
# for i in range(center_h-width_c, center_h+width_c, period):  
#     SLM[i:i + half_period,center_s - width_c:center_s+width_c] = 0 
#     SLM[center - width_c:center + width_c, i:i + half_period] = 0
# plt.imsave(f"SLM p_{period}.png", SLM,cmap='gray')

# SLM_fft=fftshift(fft2(SLM))
# SLM_mag=abs(SLM_fft)/np.max(abs(SLM_fft))
# plt.imsave(f"FFT_SLM p_{period}.png", SLM_mag,cmap='hot')
# Display the result
# Visualize the real and imaginary parts of the phase grating
plt.figure()
plt.imshow(np.real(phase_grating), cmap='gray')
plt.title('Real part of the Phase Grating')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.imag(phase_grating), cmap='gray')
plt.title('Imaginary part of the Phase Grating')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fft_magnitude_cropped, cmap='hot')  # Use log to enhance visibility of low values
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(amplitude, cmap='hot')  # Use log to enhance visibility of low values
plt.colorbar()
plt.show()