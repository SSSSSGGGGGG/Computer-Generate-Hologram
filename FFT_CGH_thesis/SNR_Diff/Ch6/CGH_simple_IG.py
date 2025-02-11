# -*- coding: utf-8 -*-
"""
Created on Monday February 10 2025
@author: Shang Gao, Ignacio Moreno

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
# nimport os
# import sys
import time

start_t=time.time()
filename="CircleR100"  
im=plt.imread(f"{filename}.png")

height=im.shape[0] # This gives the height of variable 'im'
width=im.shape[1] # This gives the with of variable 'im'

Field=np.sqrt(im[:,:,0]) # The red channel is selected only
rand = np.random.uniform(0, 1, (height , width)) # Generates random umbers [0-1]

#----------------------------------
Options = ["No noise", "Amplitude Noise", "Binary Amplitude Noise", "Phase Noise"]
for i, option in enumerate (Options,1):
    print(f"{i}. {option}")
    
Selection = int(input("Select an option (1-4): "))
    
if Selection == 1:
    Field = Field
    print("Option 1: No noise")
    
elif Selection == 2:
    Field = Field * rand  # Random Continuous Amplitude
    print("Option 2: Amplitude Noise")
    
elif Selection == 3:
    threshold = 0.5
    binarized_rand = (rand > threshold).astype(np.uint8)
    Field = Field * binarized_rand  # Random Continuous Amplitude
    print("Option 3: Binary Amplitude Noise")
    
elif Selection == 4:
    rand_2pi = 2 * np.pi * rand  # Full phase range [0, 2π]
    Field = Field * np.exp(1j * rand_2pi)  # Complex exponential
    print("Option 4: Phase Noise")
else:
    print("Non valid selection")

# """
#----------------------------------
# FOURIER TRANSFORM HOLOGRAM
Ft_Field = fftshift(fft2(Field))  
Hologram_Phase = np.angle(Ft_Field)
Hologram_Magnitude = np.abs(Ft_Field)

#----------------------------------
# SAVE FOURER TANSFORM HOLOGRAM
# PHASE
im_Phase = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 1))
im_Phase = 255 * (Hologram_Phase + np.pi)/(2*np.pi) # Converts  [+pi,-pi] to [255,0]
I=im_Phase.astype(np.uint8) # Converts into integer values into variable 'I'
IFile=Image.fromarray(I)
IFile.save(f"{filename} Phase FT.png") # Save reconstruction results.

# MAGNITUDE
Mag_Max = np.max(Hologram_Magnitude);
im_Hologram_Magnitude = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 1))
im_Hologram_Magnitude = 255*Hologram_Magnitude/Mag_Max
I=im_Hologram_Magnitude.astype(np.uint8) # Converts into integer values into variable 'I'
IFile=Image.fromarray(I)
IFile.save(f"{filename} Magnitude FT.png") # Save reconstruction results.

#----------------------------------
# HOLOGRAM RECONSTRUCTION
Rec = ifft2(ifftshift(np.exp(1j * Hologram_Phase)))
Rec_Intensity = (np.abs(Rec))**2

# SAVE HOLOGRAM RECONSTRUCTION
Rec_Max = np.max(Rec_Intensity);
im_Rec = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 1))
im_Rec = 255*Rec_Intensity/Rec_Max
I=im_Rec.astype(np.uint8) # Converts into integer values into variable 'I'
IFile=Image.fromarray(I)
IFile.save(f"{filename}1 Reconstruction.png") # Save reconstruction results.

#----------------------------------
# PLOT A PROFILE
row = 511
profile = im_Rec[row, :]

# Plot the profile
plt.figure(figsize=(10, 4))
plt.plot(profile, label=f'Row {row}')
plt.xlabel('Column Index')
plt.ylabel('Intensity Value')
plt.title('Intensity Profile Along Row')
plt.legend()
plt.show()

end_t=time.time()
print(f"Time consuming {end_t-start_t}s")