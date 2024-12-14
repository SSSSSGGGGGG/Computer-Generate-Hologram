# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:55:01 2024

@author: gaosh
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift

im_GS=plt.imread("C:/Users/gaosh/Documents/python/Computer-Generate-Hologram/FFT of imgs/GS/3D_dear_fit_sh_1_C_noL.png")
im_MP=plt.imread("C:/Users/gaosh/Documents/python/Computer-Generate-Hologram/Wirting_Holography/3D_dear_fit_sh_10_C_noL.png")
im_GS_R=im_GS[:,:,0]*255
im_MP_R=im_MP[:,:,0]*255

im_GS_R_angle=(im_GS_R/255)*1.8*np.pi
im_MP_R_angle=(im_MP_R/255)*1.8*np.pi

current_field_GS = ifft2(ifftshift(np.exp(1j * im_GS_R_angle)))
current_field_MP = ifft2(ifftshift(np.exp(1j * im_MP_R_angle)))

plt.figure()
# Reconstructed amplitude
plt.title("GS")
plt.imshow(np.abs(current_field_GS), cmap="Reds")
plt.colorbar()
plt.show()

plt.figure()
# Reconstruction error
# error = np.abs(np.abs(im_shift) - final_field)
plt.title("MP")
plt.imshow(np.abs(current_field_MP), cmap="Reds")
plt.colorbar()
plt.show()