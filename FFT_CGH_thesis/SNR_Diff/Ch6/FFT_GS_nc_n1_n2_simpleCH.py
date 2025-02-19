# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:45:57 2024

@author: gaosh
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os
import time


start_t=time.time()

os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6")
filename="One circle_1024"  
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]

randP=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*randP
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)

l=300  # Length that is from edge to center.
c_w,c_h=width//2,height//2 # Center
lh,lw=height-2*l,width-2*l  # Height and width for the window.

#R channel
im_shift_r=fftshift(im[:,:,0])
#G channel
im_shift_g=fftshift(im[:,:,1])
#B channel
im_shift_b=fftshift(im[:,:,2])


# Random noise generation.
n=1
rand=np.random.uniform(0, n, (height, width))

im_r_rand=np.sqrt(im_shift_r)*exp_rand
im_g_rand=np.sqrt(im_shift_g)*exp_rand
im_b_rand=np.sqrt(im_shift_b)*exp_rand

nw_field_r =fftshift(fft2(im_r_rand))
nw_field_g =fftshift(fft2(im_g_rand))
nw_field_b =fftshift(fft2(im_b_rand))

# In the window, the amplitudes are from original. Otherwise, the amplitudes are from retrieved amplitudes in inverse FFT2 computation.
im_r_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]
im_g_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]
im_b_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]

im_r_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
im_g_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
im_b_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]

im_r_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
im_g_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
im_b_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]

# Compute forward 2D Fast Fourier Transform (FFT2) of each channel.
w_field_r =fftshift(fft2(im_r_rand))
w_field_g =fftshift(fft2(im_g_rand))
w_field_b =fftshift(fft2(im_b_rand))

# Inverse FFT2 computation that is the same as previous iterative loop.
w_field_r_if = ifft2(ifftshift(np.exp(1j * np.angle(w_field_r))))
w_field_g_if = ifft2(ifftshift(np.exp(1j * np.angle(w_field_g))))
w_field_b_if = ifft2(ifftshift(np.exp(1j * np.angle(w_field_b))))

# Inverse FFT2 computation that is the same as previous iterative loop.
nw_field_r_if = ifft2(ifftshift(np.exp(1j * np.angle(nw_field_r))))
nw_field_g_if = ifft2(ifftshift(np.exp(1j * np.angle(nw_field_g))))
nw_field_b_if = ifft2(ifftshift(np.exp(1j * np.angle(nw_field_b))))

plt.figure()
plt.imshow(abs(nw_field_r), cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(abs(w_field_r), cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.angle(nw_field_r), cmap="hsv")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.angle(w_field_r), cmap="hsv")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fftshift(abs(nw_field_r_if)), cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fftshift(abs(w_field_r_if)), cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fftshift(abs(nw_field_r_if)**2), cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(fftshift(abs(w_field_r_if)**2), cmap="hot")
plt.colorbar()
plt.show()