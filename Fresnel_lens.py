# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt

def fresnel_lens_phase_mask(width, height, focal_length, wavelength, pixel_size):
    """
    Generate a phase mask for a Fresnel lens.
    
    Parameters:
    width: int
        The width of the SLM or phase mask (number of pixels).
    height: int
        The height of the SLM or phase mask (number of pixels).
    focal_length: float
        The focal length of the Fresnel lens (in meters).
    wavelength: float
        The wavelength of the light (in meters).
    pixel_size: float
        The size of each pixel (in meters).
    
    Returns:
    phase_mask: 2D array
        The Fresnel lens phase mask, scaled to 0-2pi for SLM use.
    """
    # Create a coordinate grid (X, Y) centered at the middle of the SLM
    x = np.linspace(-width//2, width//2, width) * pixel_size
    y = np.linspace(-height//2, height//2, height) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    # Calculate radial distance from the center of the SLM
    R = np.sqrt(X**2 + Y**2)
    
    # Fresnel phase shift: quadratic term based on radial distance
    phase_shift = (np.pi / wavelength / focal_length) * (R**2)
    
    # Wrap the phase shift into the range 0 to 2pi
    phase_mask = np.mod(phase_shift, 2 * np.pi)
    
    return phase_mask

# Parameters for the Fresnel lens
width, height = 1920, 1080  # Size of the phase mask (number of pixels)
focal_length = 1  # Focal length of the Fresnel lens (in meters)
wavelength = 532e-9  # Wavelength of light (532 nm = green laser)
pixel_size = 4.5e-6  # Size of each pixel (4.5 micrometers)

# Generate the Fresnel lens phase mask
fresnel_phase_mask = fresnel_lens_phase_mask(width, height, focal_length, wavelength, pixel_size)

# Visualize the Fresnel lens phase mask
plt.imshow(fresnel_phase_mask, cmap='twilight', interpolation='nearest')
plt.colorbar()
plt.title("Fresnel Lens Phase Mask")
plt.show()

