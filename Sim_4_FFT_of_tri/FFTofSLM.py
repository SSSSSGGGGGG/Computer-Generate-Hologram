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
period=64*4.6e-6
half_period=int(period/2)
phase=0.637*np.pi
period_X=92*4.6e-6
period_Y=400*4.6e-6

x = np.linspace(-0.5*width*4.6e-6, 0.5*width*4.6e-6, width)
y = np.linspace(-0.5*height*4.6e-6, 0.5*height*4.6e-6, height)
X, Y = np.meshgrid(x, y)

binary_phase = np.where((X // period) % 2 == 0, 1, np.exp(1j *phase))

blazed_phase_X = ((2  / period_X) * ((X) %  (period_X)))
blazed_phase_offset_X=blazed_phase_X
blazed_grating_X = np.exp(1j *blazed_phase_offset_X*np.pi)#np.mod(blazed_phase, 2 * np.pi)

blazed_phase_Y = ((2 / period_Y) * ((Y) %  (period_Y)))
blazed_phase_offset_Y=blazed_phase_Y
blazed_grating_Y = np.exp(1j *blazed_phase_offset_Y*np.pi)

beam_width = 200*4.6e-6
Guassia_amplitude = np.exp(-((X - 0*4.6e-6)**2 + (1*Y - 0)**2) / (2 * beam_width**2))
Guassia_amplitude=Guassia_amplitude/np.max(Guassia_amplitude)

fft_grating_binary = fftshift(fft2(binary_phase*Guassia_amplitude))
I_binary=np.abs(fft_grating_binary)**2
sumI=np.sum(I_binary)
I_binary_nor=I_binary/sumI
I_max=np.max(I_binary)
Index_max = np.unravel_index(np.argmax(I_binary), I_binary.shape)

fft_grating_withBlazed = fftshift(fft2(blazed_grating_X*blazed_grating_Y*binary_phase*Guassia_amplitude))
I_withBlazed=np.abs(fft_grating_withBlazed)**2
sumI_withBlazed=np.sum(I_withBlazed)
I_withBlazed_nor=I_withBlazed/sumI_withBlazed

# Define the center region to crop
center_h = height // 2
center_w = width // 2
crop_size_h = 50  # Size of the cropped region (200x200 pixels)
crop_size_w = 200
# Crop the FFT magnitude
start_h = center_h - crop_size_h // 2
end_h = center_h + crop_size_h // 2
start_w = center_w - crop_size_w // 2
end_w = center_w + crop_size_w // 2
I_cropped_binary = I_binary_nor[start_h:end_h, start_w:end_w]
I_cropped_withBlazed = I_withBlazed_nor[start_h:end_h, start_w:end_w]

frequencies_x = np.fft.fftshift(np.fft.fftfreq(width, d=1))
frequencies_y = np.fft.fftshift(np.fft.fftfreq(height, d=1))
# print(I_max,sumI)
# plt.figure()
# plt.imshow(np.real(binary_phase), cmap='gray')
# plt.title('Real part of the binary phase grating')
# plt.colorbar()
# plt.show()


plt.figure()
plt.imshow(Guassia_amplitude, cmap='hot')
plt.title('Beam')
plt.colorbar()
plt.imsave(f"Gaussian Beam width {beam_width}m.png",Guassia_amplitude,cmap="hot")

plt.figure()
plt.imshow(I_cropped_binary, cmap='hot')  
plt.title('FFT of the binary phase grating * Beam')
plt.colorbar()
plt.show()
plt.imsave(f"FFT of binary phase grating p_{period/4.6e-6}.png",I_cropped_binary,cmap="hot")

plt.figure()
plt.imshow(I_cropped_withBlazed, cmap='hot')  
plt.title('FFT of compensated binary phase grating * Beam')
plt.colorbar()
plt.show()
plt.imsave(f"FFT of compensated binary phase grating pV_{period_Y/4.6e-6} pH_{period_X/4.6e-6}.png",I_cropped_withBlazed,cmap="hot")
# plt.figure()
# plt.imshow(I_cropped_binary, cmap='hot')  
# plt.title('FFT of the binary phase grating')

# # Create x and y tick positions for cropped image
# # num_ticks = 10  # Number of ticks
# # xtick_positions = np.linspace(0, I_cropped_binary.shape[1] - 1, num_ticks, dtype=int)
# # ytick_positions = np.linspace(0, I_cropped_binary.shape[0] - 1, num_ticks, dtype=int)
# # # Corresponding frequency labels for ticks
# # xtick_labels = np.round(frequencies_x[start_w:end_w][xtick_positions], 2)
# # ytick_labels = np.round(frequencies_y[start_h:end_h][ytick_positions], 2)

# # # Set x and y ticks to display spatial frequencies
# # plt.xticks(xtick_positions, xtick_labels)
# # plt.yticks(ytick_positions, ytick_labels)
# plt.colorbar()
# plt.show()