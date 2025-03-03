# -*- coding: utf-8 -*-
"""
Created on February 13 2025
@author: Shang Gao, Ignacio Moreno

Simulates the temporal integration of multiple CGHs

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import time

start_t=time.time()

N = 1024 # Image array size
PI = np.pi

Target = np.load('Target.npy')

Iterations = 1

Intensity = np.zeros((1024, 1024), dtype=float)  # Sets Intensity to zero

count = 0
while count < Iterations:
    print(count)
    count += 1
    #----------------------------------
    # Creates the array of random numbers [0-1]
    RandomImage = Image.new("L", (N,N), "black") # Defines a gray scale image
    Rand1 = np.random.uniform(0,1,(N,N)) 
    RandomPhase = np.exp(1j*2*PI*Rand1)

    Field = Target * RandomPhase
    #----------------------------------
    # FOURIER TRANSFORM
    FT_Field = ifftshift(ifft2(ifftshift(Field)))  
    FT_Magnitude = np.abs(FT_Field)
    FT_Phase = np.angle(FT_Field)
    #----------------------------------
    # MAGNITUDE = 1
    FT_Field = np.exp(1j*FT_Phase)
    #----------------------------------
    # INVERSE FOURIER TRANSFORM
    Field = fftshift(fft2(fftshift(FT_Field)))  
    Magnitude = np.abs(Field)
    Intensity = Intensity + np.square(Magnitude)
   
    #----------------------------------
    # Plots the magnitude and the intensity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.imshow(Magnitude, cmap='hot')
    ax1.axis('off')
    im2 = ax2.imshow(Intensity, cmap='hot')
    ax2.axis('off')
    plt.show()
    #---------------------------------- END OF THE LOOP

# Saves the images at the end
plt.imsave("Rec_Intensity.bmp", Intensity, cmap='hot')
plt.imsave("Rec_Magnitude.bmp", Magnitude, cmap='hot')

#----------------------------------
# PLOT A PROFILE
row = 511
profile = Intensity[row,384:640]

# Plot the profile
plt.figure(figsize=(10, 5))
plt.plot(profile, linewidth=5)
# plt.xlabel('Column Index')
# plt.ylabel('Intensity Value')
# plt.title('Intensity Profile Along Row')
# plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()
#----------------------------------
end_t=time.time()
print(f"Time consuming {end_t-start_t}s")