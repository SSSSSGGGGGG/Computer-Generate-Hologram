# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:21:55 2025

@author:Shang Gao 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

h, w=256,256 # Size of hologram
c=h//2 # Center

""" A binary phase grating"""
BG=np.ones((h, w),dtype="complex128")
Phase_diff=np.pi  # This is the phase difference that can be changed
p=16 # period
for i in range(0,w+1,p*2):
    for j in range(0,h):
        BG[j,i:i+p]=0
        binary_phase=BG*np.exp(1j*Phase_diff) # There are two phases in one period: 0, and "Phase_diff"
plt.figure()
plt.imshow(abs(binary_phase), cmap="gray")
plt.title("Binary Phase Grating")
plt.colorbar()
plt.show()

""" Fouier transform of the binary phase grating"""
Diffraction_pattern=fftshift(fft2(binary_phase)) # Diffraction orders separated by binary grating

plt.figure()
plt.imshow(abs(Diffraction_pattern), cmap="hot")
plt.title("Relization of Binary Phase Grating")
plt.colorbar()
plt.show()

""" A spiral phase mask"""
y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
angles = np.arctan2(y - c, x - c)  # Angle range [-pi, pi]
positive = (angles + np.pi) # Now in [0, 2*pi]
Spiral_phase = positive  # Along azimuthal the phase changes from 0 to 2pi

plt.figure()
plt.imshow(Spiral_phase, cmap='gray')
plt.colorbar()
plt.title('A spiral phase mask')
plt.show()

""" Generation of Gaussian Beam"""
x_G = np.linspace(-0.5*w, 0.5*w, w)
y_G = np.linspace(-0.5*h, 0.5*h, h)
X, Y = np.meshgrid(x_G, y_G)
beam_width = 10 # Width of the beam profile
Guassia_amplitude = np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * beam_width**2))
Guassia_amplitude=Guassia_amplitude/np.max(Guassia_amplitude)

plt.figure()
plt.imshow(Guassia_amplitude, cmap="hot")
plt.title("Gaussian Beam Profile")
plt.colorbar()
plt.show()

""" Fouier transform of the spiral phase mask"""
Diffraction_pattern_S=fftshift(fft2(Guassia_amplitude*np.exp(1j*Spiral_phase))) # Spiral beam is multiplied by a Gaussian beam to magnify the vortex beam
# The beam is shaped in to vortex (donut-shaped)
plt.figure()
plt.imshow(abs(Diffraction_pattern_S), cmap="hot")
plt.title("Relization of spiral phase mask - Vortex Beam")
plt.colorbar()
plt.show()

""" Superposition of vortex and binary grating"""
Diffraction_pattern_C=fftshift(fft2(binary_phase*np.exp(1j*Spiral_phase))) # Spiral beam is multiplied by the binary phase grating
plt.figure()
plt.imshow(abs(Diffraction_pattern_C), cmap="hot") # Convolution between diffraction orders and vortex beam
plt.title("Superposition of vortex and binary grating")
plt.colorbar()
plt.show()
