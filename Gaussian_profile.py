# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
width,height=1920,1080  # Size of the mask
beam_width = 10  # Width of the Gaussian beam
center = (0, 0)  # Center of the Gaussian profile
x = np.linspace(-width//2, width//2, width)
y = np.linspace(-height//2, height//2, height)
X, Y = np.meshgrid(x, y)

# Create Gaussian amplitude profile
amplitude = np.exp(-((X - center[0])**2 + (Y - center[1])**2) / (2 * beam_width**2))
amplitude_nor =amplitude/amplitude.max()

phase_mask = np.log(amplitude_nor)

# Visualize the mask
plt.figure(1)
plt.imshow(amplitude_nor, cmap='hot')
plt.colorbar()
plt.title("Gaussian Beam Profile")
plt.show()

plt.figure(2)
plt.imshow(abs(phase_mask), cmap='hot')
plt.colorbar()
plt.title("phase_mask")
plt.show()