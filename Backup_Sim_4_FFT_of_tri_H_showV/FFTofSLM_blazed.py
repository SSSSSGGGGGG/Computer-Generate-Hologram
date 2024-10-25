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
p_X=92
max_phase=2
p_Y=400
period_X=p_X*4.6e-6
period_Y=p_Y*4.6e-6
delta_X=1/p_X*width
delta_Y=1/p_Y*height
print(f"fx1={delta_X}")


x = np.linspace(-0.5*width*4.6e-6, 0.5*width*4.6e-6, width)
y = np.linspace(-0.5*height*4.6e-6, 0.5*height*4.6e-6, height)
# x = np.linspace(-0.5*width, 0.5*width, width)
# y = np.linspace(-0.5*height, 0.5*height, height)
X, Y = np.meshgrid(x, y)
# Create a sinusoidal phase grating
blazed_phase_X = ((max_phase  / period_X) * ((X) %  (period_X)))
blazed_phase_offset_X=blazed_phase_X
blazed_grating_X = np.exp(1j *blazed_phase_offset_X*np.pi)#np.mod(blazed_phase, 2 * np.pi)

blazed_phase_Y = ((max_phase / period_Y) * ((Y) %  (period_Y)))
blazed_phase_offset_Y=blazed_phase_Y
blazed_grating_Y = np.exp(1j *blazed_phase_offset_Y*np.pi)

beam_width = 150*4.6e-6
Guassia_amplitude = np.exp(-((X - 0*4.6e-6)**2 + (Y - 0)**2) / (2 * beam_width**2))
Guassia_amplitude=Guassia_amplitude/np.max(Guassia_amplitude)

fft_grating_X_G = fftshift(fft2(blazed_grating_X*Guassia_amplitude))
I_X_G = np.abs(fft_grating_X_G)**2
I_X_G =I_X_G/np.sum(I_X_G)

fft_grating_X = fftshift(fft2(blazed_grating_X))
I_X = np.abs(fft_grating_X)**2
I_X =I_X/np.sum(I_X)
print(f"position for 1st and 2nd {int(width/2+delta_X)-960,int(width/2+2*delta_X)-960}")
print(f"maximum={np.max(I_X)},0th_X={I_X[int(height/2),int(width/2)]}")
print(f"1th_X={I_X[int(height/2),int(width/2+delta_X)]},2nd_X={I_X[int(height/2),int(width/2+2*delta_X)]}")
center_h = height // 2
center_w = width // 2
crop_size = 200  # Size of the cropped region (200x200 pixels)

# Crop the FFT magnitude
start_h = center_h - crop_size // 2
end_h = center_h + crop_size // 2
start_w = center_w - crop_size // 2
end_w = center_w + crop_size // 2
fft_magnitude_cropped = I_X_G[start_h:end_h, start_w:end_w]

# plt.figure()
# plt.imshow(Guassia_amplitude, cmap='hot')
# plt.title('beam')
# plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(I_X_G, cmap='hot')
plt.title(f'blazed_phase')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fft_magnitude_cropped, cmap='hot')  # Use log to enhance visibility of low values
plt.colorbar()
plt.show()
plt.imsave(f"FFT of blazed p={p_X,max_phase}pi.png",fft_magnitude_cropped, cmap='hot')
