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
p=48 #half period
phase = 0.64 * np.pi 
G1 = np.exp(1j * phase)
G2 = np.exp(1j * 0)  
delta_bi=0.5*height/p

p_X=92#79#81
p_Y=1080#400#640
delta_X = 1 / p_X * width
delta_Y = 1 / p_Y * height
max_phase=2*np.pi

x = np.linspace(-0.5*width, 0.5*width, width)
y = np.linspace(-0.5*height, 0.5*height, height)
X, Y = np.meshgrid(x, y)
 
"""Binary""" 
binary_phase = np.ones((height, width),dtype="complex128")
stripe_width = p  # period
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

"""Blazed""" 
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
"""Beam""" 
beam_width = 350
Guassia_amplitude = np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * beam_width**2))
Guassia_amplitude=Guassia_amplitude/np.max(Guassia_amplitude)
"""FFT of binary""" 
fft_grating_binary = fftshift(fft2(binary_phase))#*Guassia_amplitude
I_binary=np.abs(fft_grating_binary)**2
sumI=np.sum(I_binary)
I_binary_nor=I_binary/sumI
plt.imsave(f"FFT of binary phase grating p_{2*p}px no beam.png",I_binary_nor,cmap="hot")

I_0=I_binary_nor[540,960]
I_1=I_binary_nor[540+int(delta_bi),960]
I_n1=I_binary_nor[540-int(delta_bi),960]
print(f"Binary: f1={delta_bi}, I_1:{540+int(delta_bi)},I_n1:{540-int(delta_bi)}")
print(f"I_0={I_0},I_1={I_1},I_-1={I_n1} ")

fft_grating_binary_G = fftshift(fft2(binary_phase*Guassia_amplitude))#*Guassia_amplitude
I_binary_G=np.abs(fft_grating_binary_G)**2
sumI_G=np.sum(I_binary_G)
I_binary_G_nor=I_binary_G/sumI_G
I_0_G=I_binary_G_nor[540,960]
I_1_G=I_binary_G_nor[540+int(delta_bi),960]
I_n1_G=I_binary_G_nor[540-int(delta_bi),960]
# print(f"With beam, I_0={I_0_G},I_1={I_1_G},I_-1={I_n1_G} ")
"""FFT of binary*blazed""" 
fft_grating_withBlazed = fftshift(fft2(blazed_grating_X
                                       *binary_phase))
I_withBlazed=np.abs(fft_grating_withBlazed)**2
sumI_withBlazed=np.sum(I_withBlazed)
I_withBlazed_nor=I_withBlazed/sumI_withBlazed

I_0_b=I_withBlazed_nor[540+1+int(1080/p_Y),960+int(1920/p_X)]
I_1_b=I_withBlazed_nor[540+1+int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
I_n1_b=I_withBlazed_nor[540-int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
print(f"Blazed: fy={1080/p_Y},fx={1920/p_X},I_0_x:{540+int(1080/p_Y)},I_0_y:{960+int(1920/p_X)}")
print(f"I_0={I_0_b},I_1={I_1_b},I_-1={I_n1_b}, CE={I_0_b/I_0} ")
fft_grating_withBlazed_G = fftshift(fft2(blazed_grating_X*binary_phase*Guassia_amplitude))#*blazed_grating_Y
I_withBlazed_G=np.abs(fft_grating_withBlazed_G)**2
sumI_withBlazed_G=np.sum(I_withBlazed_G)
I_withBlazed_G_nor=I_withBlazed_G/sumI_withBlazed_G
I_0_b_G=I_withBlazed_G_nor[540+int(1080/p_Y),960+int(1920/p_X)]
I_1_b_G=I_withBlazed_G_nor[540+int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
I_n1_b_G=I_withBlazed_G_nor[540-int(delta_bi)+int(1080/p_Y),960+int(1920/p_X)]
# print(f"With beam,{960+int(1920/p_X)}, I_0={I_0_b_G},I_1={I_1_b_G},I_-1={I_n1_b_G}, CE={I_0_b_G/I_0_G} ")
"""Define the center region to crop"""
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
"""Built-in spatial frequency"""
frequencies_x = np.fft.fftshift(np.fft.fftfreq(width, d=1))
frequencies_y = np.fft.fftshift(np.fft.fftfreq(height, d=1))
# print(I_max,sumI)
"""Show"""
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
plt.imsave(f"FFT of all gratings pV_{p_Y}px pH_{p_X}px {max_phase/np.pi}pi.png",I_cropped_withBlazed,cmap="hot")
