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


os.chdir("C:/Users/Laboratorio/MakeHologram/FFT of imgs")
filename="3planets_h20"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]
im_shift=fftshift(im[:,:,0])

im_in=ifftshift(im_shift)
im_rgb_ft=fft2(im_shift)
phase_rgb = np.angle(im_rgb_ft)
Reconstruction=ifft2(im_in)#np.exp(1j*phase_rgb)))
phase_in=np.angle(Reconstruction)

plt.figure()
plt.imshow(phase_rgb, label="magnitude", cmap="Reds")
plt.show()
plt.figure()
plt.imshow(phase_in, label="magnitude", cmap="Reds")
plt.show()
# rand=np.random.uniform(0, 1, (height, width))

# rand_2pi=np.pi*rand
# rand_ma=np.max(rand_2pi)
# rand_mi=np.min(rand_2pi)
# exp_rand=np.exp(1j*rand_2pi)
# im_r_shift=exp_rand*im_shift

# im_r_ft=fftshift(fft2(im_r_shift))
# phase_r = np.angle(im_r_ft)
# Reconstruction_noise=ifft2(ifftshift(np.exp(1j*phase_r)))
# plt.figure()
# plt.imshow(ifftshift(abs(Reconstruction_noise)), label="magnitude", cmap="Reds")
# plt.show()

# i_1=fftshift(fft2(exp_rand*Reconstruction_noise)) #Reconstruction_noise
# i_1_phase=np.angle(i_1)
# R_1=ifft2(ifftshift(np.exp(1j*i_1_phase)))
# plt.figure()
# plt.imshow(ifftshift(abs(R_1)), label="magnitude", cmap="Reds")
# plt.show()

# i_1_s=fftshift(fft2(exp_rand*im_shift*Reconstruction_noise)) #Reconstruction_noise
# i_1_s_phase=np.angle(i_1_s)
# R_1_s=ifft2(ifftshift(np.exp(1j*i_1_s_phase)))
# plt.figure()
# plt.imshow(ifftshift(abs(R_1_s)), label="magnitude", cmap="Reds")
# plt.show()

# i_2=fftshift(fft2(exp_rand*R_1)) #Reconstruction_noise
# i_2_phase=np.angle(i_2)
# R_2=ifft2(ifftshift(np.exp(1j*i_2_phase)))
# plt.figure()
# plt.imshow(ifftshift(abs(R_2)), label="magnitude", cmap="Reds")
# plt.show()

# i_2_s=fftshift(fft2(exp_rand*im_shift*R_1_s)) #Reconstruction_noise
# i_2_s_phase=np.angle(i_2_s)
# R_2_s=ifft2(ifftshift(np.exp(1j*i_2_s_phase)))
# plt.figure()
# plt.imshow(ifftshift(abs(R_2_s)), label="magnitude", cmap="Reds")
# plt.show()