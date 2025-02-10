# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:45:57 2024

@author: gaoshang
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os
import sys
import time

start_t=time.time()
os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6")
filename="One circle_1024"  
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]

field_r=np.sqrt(im[:,:,0])
field_g=np.sqrt(im[:,:,1])
field_b=np.sqrt(im[:,:,2]) 

# Random phase generation
rand = np.random.uniform(0, 1, (height , width))
# Convert to binary noise: 1 if >= 0.5, else 0
binary_noise = (rand >= 0.5).astype(int)
plt.imshow(binary_noise,cmap="Grays")
cbar = plt.colorbar(ticks=[0, 1])
cbar.ax.set_yticklabels(['0','1']) 
plt.axis("off")
# plt.imsave("Random noise.png",rand,cmap="Grays")
plt.show()
# if you want to apply niose inout y.
noise="y"#input(f"Do you want to employ noise (y/n):")
try:
    if noise == "y":
        # Apply noise to the filed which is the squareroot of intensity. Compute forward FT.
        current_field_r = fftshift(fft2(binary_noise * field_r))  # Red channel
        current_field_g = fftshift(fft2(binary_noise * field_g))  # Green channel
        current_field_b = fftshift(fft2(binary_noise * field_b))  # Blue channel
    elif noise == "n":
        # Without noise
        current_field_r = fftshift(fft2(field_r))  # Red channel
        current_field_g = fftshift(fft2(field_g))  # Green channel
        current_field_b = fftshift(fft2(field_b))  # Blue channel
    else:
        # Raise an exception if input is invalid
        raise ValueError("Invalid input! Please input 'y' or 'n'. Program will terminate.")
        
except ValueError as e:
    print(e)
    sys.exit("end")  

# Final optimized phase for display or application on SLM, in simulation is for saving hologram images.
magnitude_r=np.sqrt(abs(current_field_r)**2/np.max(abs(current_field_r)**2)) 
optimized_phase_r = np.angle(current_field_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/2)

magnitude_g=np.sqrt(abs(current_field_g)**2/np.max(abs(current_field_g)**2))
optimized_phase_g = np.angle(current_field_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/2)

magnitude_b=np.sqrt(abs(current_field_b)**2/np.max(abs(current_field_b)**2))
optimized_phase_b = np.angle(current_field_b)
phase_br_modi=(optimized_phase_b/np.pi+1)*(255/2)

# Save magnitude
im_modify_m = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_m[:,:,0] = magnitude_r*255
im_modify_m[:,:,1] = magnitude_g*255
im_modify_m[:,:,2] = magnitude_b*255
im_m = im_modify_m.astype(np.uint8)
im_m_ = Image.fromarray(im_m)
im_m_.save(f"{filename} Mag_BiNoise_{noise}.png")

# Save phase-only holograms.
im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi
im_modify_noL[:,:,1] = phase_gr_modi
im_modify_noL[:,:,2] = phase_br_modi
im_cropped = im_modify_noL.astype(np.uint8)
im_modi = Image.fromarray(im_cropped)
im_modi.save(f"{filename} Holo_BiNoise_{noise}.png")
# Compute inverse FT of phase, and the amplitudes are 1.
Reconstruction_holo1_r = ifft2(ifftshift(np.exp(1j * optimized_phase_r )))
Reconstruction_holo1_g = ifft2(ifftshift(np.exp(1j * optimized_phase_g )))
Reconstruction_holo1_b = ifft2(ifftshift(np.exp(1j * optimized_phase_b )))
# Adjust factor for different visual effect, if using small value we can see noise. 
factor=1#0.1#1#0.4#0.7#0.02
# Intensity of each pixel is normalized by the sum of all intensity in its channel.
I1_r=np.abs(Reconstruction_holo1_r)**2
I1_r_normalized = I1_r / np.sum(I1_r)
I1_g=np.abs(Reconstruction_holo1_g)**2
I1_g_normalized = I1_g / np.sum(I1_g)
I1_b=np.abs(Reconstruction_holo1_b)**2
I1_b_normalized = I1_b / np.sum(I1_b)
# Normalized intensites are divided by the maximum to make image of reconstruction in [0,1], mulplied by 255.
I1_rgb=np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
I1_rgb[:, :, 0] = I1_r_normalized/(np.max(I1_r_normalized)*factor)*255
I1_rgb[:, :, 1] = I1_g_normalized/(np.max(I1_g_normalized)*factor)*255
I1_rgb[:, :, 2] = I1_b_normalized/(np.max(I1_b_normalized)*factor)*255
I_tf=I1_rgb.astype(np.uint8)
Irgb=Image.fromarray(I_tf)
Irgb.save(f"{filename} Reconstruction_BiNoise_{noise}.png") # Save reconstruction results.

end_t=time.time()
print(f"Time consuming {end_t-start_t}s")