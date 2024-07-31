# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image
import numpy as np
import scipy as sp
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import matplotlib.pyplot as plt
import os
from skimage import color


os.chdir("C:/Users/Laboratorio/MakeHologram/OriginalImage")
filename="1.jpg"
im=plt.imread(filename)[:,:,:3]
shift_state="shift" # to check the shift influence,

im_r=im[:,:,0]*255

im_g=im[:,:,1]

im_b=im[:,:,2]



im_r_process=(im_r-1)*(-1)*255
# im_r_fft=fftshift(fft2(im_r))
im_r_fft=fftshift(fft2(im_r_process))

# rows, cols = im_r.shape
# # Frequency coordinates
# u = np.fft.fftshift(np.fft.fftfreq(rows))  # Frequency coordinates along rows
# v = np.fft.fftshift(np.fft.fftfreq(cols))  # Frequency coordinates along columns
# # print(u,v)
# # Create frequency grid
# U, V = np.meshgrid(u, v)

#(-pi, pi]
phase = np.angle(im_r_fft)
# the calibration between red and gray level 1.85-->255
# interval=255/(2*np.pi)
angle_0_pi = phase*(510/(2*np.pi))

# angle_0_pi =angle_0_pi # this is the one in 0-255

magnitude = np.abs(im_r_fft)

combined = np.exp(1j * phase)
im_r_if = np.fft.ifft2(combined)

# I_r=im_r_if*im_r_if.conjugate()
plt.figure(1)
plt.imshow(np.abs(im_r_if),cmap="hot")
plt.show()
plt.figure(2)
plt.imshow(im_r,cmap="hot")
plt.show()

im_new=np.zeros_like(im)
im_new[:,:,0]=np.mod(angle_0_pi,255)
im_new_array = im_new.astype(np.uint8)

im_new_t = Image.fromarray(im_new_array)
im_new_t.save(f"ft of {filename} {shift_state}.png")
# im_new_t.show()

