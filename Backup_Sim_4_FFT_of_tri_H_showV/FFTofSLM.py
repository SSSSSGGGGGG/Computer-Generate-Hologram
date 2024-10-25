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
p=64 #half period
max_phase=1.85
delta_bi=0.5*height/p
period=p*4.6e-6
half_period=int(period/2)
phase=0.64*np.pi
p_X=79#92
p_Y=400#640
period_X=p_X*4.6e-6
period_Y=p_Y*4.6e-6


x = np.linspace(-0.5*width*4.6e-6, 0.5*width*4.6e-6, width)
y = np.linspace(-0.5*height*4.6e-6, 0.5*height*4.6e-6, height)
X, Y = np.meshgrid(x, y)

binary_phase = np.where((Y // period) % 2 == 0, 1, np.exp(1j *phase)) # reminder should be 1 or 0 for %2

blazed_phase_X = ((max_phase  / period_X) * ((X) %  (period_X)))
blazed_phase_offset_X=blazed_phase_X
blazed_grating_X = np.exp(1j *blazed_phase_offset_X*np.pi)#np.mod(blazed_phase, 2 * np.pi)

blazed_phase_Y = ((max_phase / period_Y) * ((Y) %  (period_Y)))
blazed_phase_offset_Y=blazed_phase_Y
blazed_grating_Y = np.exp(1j *blazed_phase_offset_Y*np.pi)

beam_width = 150*4.6e-6
Guassia_amplitude = np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * beam_width**2))
Guassia_amplitude=Guassia_amplitude/np.max(Guassia_amplitude)

fft_grating_binary_G = fftshift(fft2(binary_phase*Guassia_amplitude))#*Guassia_amplitude
I_binary_G=np.abs(fft_grating_binary_G)**2
sumI_G=np.sum(I_binary_G)
I_binary_G_nor=I_binary_G/sumI_G

fft_grating_binary = fftshift(fft2(binary_phase))#*Guassia_amplitude
I_binary=np.abs(fft_grating_binary)**2
sumI=np.sum(I_binary)
I_binary_nor=I_binary/sumI
plt.imsave(f"FFT of binary phase grating p_{2*p}px no beam.png",I_binary_nor,cmap="hot")
# I_max=np.max(I_binary)
# Index_max = np.unravel_index(np.argmax(I_binary), I_binary.shape)

I_0=I_binary_nor[540,960]
I_1=I_binary_nor[540+int(delta_bi),960]
I_n1=I_binary_nor[540-int(delta_bi),960]
print(f"f1={delta_bi}, I_1={540+int(delta_bi)},I_1={540-int(delta_bi)}")
print(f"I_0={I_0},I_1={I_1},I_-1={I_n1} ")

I_0_G=I_binary_G_nor[540,960]
I_1_G=I_binary_G_nor[540+int(delta_bi),960]
I_n1_G=I_binary_G_nor[540-int(delta_bi),960]
print(f"With beam, I_0={I_0_G},I_1={I_1_G},I_-1={I_n1_G} ")

fft_grating_withBlazed_G = fftshift(fft2(blazed_grating_X*binary_phase*Guassia_amplitude))#*blazed_grating_Y
I_withBlazed_G=np.abs(fft_grating_withBlazed_G)**2
sumI_withBlazed_G=np.sum(I_withBlazed_G)
I_withBlazed_G_nor=I_withBlazed_G/sumI_withBlazed_G

fft_grating_withBlazed = fftshift(fft2(blazed_grating_X*binary_phase))
I_withBlazed=np.abs(fft_grating_withBlazed)**2
sumI_withBlazed=np.sum(I_withBlazed)
I_withBlazed_nor=I_withBlazed/sumI_withBlazed

I_0_b=I_withBlazed_nor[540+int(1080/p_Y),960+int(1920/p_X)]
I_1_b=I_withBlazed_nor[540+int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
I_n1_b=I_withBlazed_nor[540-int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
print(f"{540+int(1080/p_Y)},{960+int(1920/p_X)}, I_0={I_0_b},I_1={I_1_b},I_-1={I_n1_b}, CE={I_0_b/I_0} ")

I_0_b_G=I_withBlazed_G_nor[540+int(1080/p_Y),960+int(1920/p_X)]
I_1_b_G=I_withBlazed_G_nor[540+int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
I_n1_b_G=I_withBlazed_G_nor[540-int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
print(f"With beam,{960+int(1920/p_X)}, I_0={I_0_b_G},I_1={I_1_b_G},I_-1={I_n1_b_G}, CE={I_0_b_G/I_0_G} ")
# Define the center region to crop
center_h = height // 2
center_w = width // 2
crop_size_h = 240  # Size of the cropped region (200x200 pixels)
crop_size_w = 100
# Crop the FFT magnitude
start_h = center_h - crop_size_h // 2
end_h = center_h + crop_size_h // 2
start_w = center_w - crop_size_w // 2
end_w = center_w + crop_size_w // 2
I_cropped_binary = I_binary_G_nor[start_h:end_h, start_w:end_w]
I_cropped_withBlazed = I_withBlazed_G_nor[start_h:end_h, start_w:end_w]

frequencies_x = np.fft.fftshift(np.fft.fftfreq(width, d=1))
frequencies_y = np.fft.fftshift(np.fft.fftfreq(height, d=1))
# print(I_max,sumI)
plt.figure()
plt.imshow(np.real(binary_phase), cmap='gray')
plt.title('Real part of the binary phase grating')
plt.colorbar()
plt.show()
# plt.imsave(f"Binary phase grating p_{period/4.6e-6}.png",binary_phase,cmap="gray")

plt.figure()
plt.imshow(blazed_phase_X, cmap='gray')
plt.title('Blazed phase grating_H')
plt.colorbar()
plt.show()
# plt.imsave(f"Blazed phase grating pH_{period_X/4.6e-6}.png",blazed_grating_X,cmap="gray")

plt.figure()
plt.imshow(blazed_phase_Y, cmap='gray')
plt.title('Blazed phase grating_V')
plt.colorbar()
plt.show()
# plt.imsave(f"Blazed phase grating pV_{period_Y/4.6e-6}.png",blazed_grating_Y,cmap="gray")

# plt.figure()
# plt.imshow(Guassia_amplitude, cmap='hot',vmin=0, vmax=1)
# plt.title('Beam')
# plt.colorbar()
# plt.imsave(f"Gaussian Beam width {beam_width/4.6e-6}px.png",Guassia_amplitude,cmap="hot")

plt.figure()
plt.imshow(I_cropped_binary, cmap='hot')  #,vmin=0, vmax=0.2
plt.title('FFT of the binary phase grating * Beam')
plt.colorbar()
# plt.text(0, 100, f"I_0={I_0},I_1={I_1},I_-1={I_n1}")
plt.show()
plt.imsave(f"FFT of binary phase grating p_{2*p}px.png",I_cropped_binary,cmap="hot")

plt.figure()
plt.imshow(I_cropped_withBlazed, cmap='hot')  #,vmin=0, vmax=0.2
plt.title('FFT of compensated binary phase grating * Beam')
plt.colorbar()
plt.show()
plt.imsave(f"FFT of X compensated binary phase grating pV_{period_Y/4.6e-6}px pH_{period_X/4.6e-6}px.png",I_cropped_withBlazed,cmap="hot")
