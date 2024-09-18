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
filename="Elephant_"
im=plt.imread(f"{filename}.png")
height=im.shape[0]
width=im.shape[1]
# holo_ex=plt.imread("Hologram (2).tiff")
# holo_ex_r=holo_ex[:,:,:3]
im_shift=fftshift(im[:,:,0])
im_rgb_ft=fftshift(fft2(im_shift))
phase_rgb = np.angle(im_rgb_ft)

phase_new=phase_rgb.astype(np.uint8)
phase_save=Image.fromarray(phase_new)
plt.imsave(f"{filename}_mono.png", phase_rgb,cmap='gray')

plt.figure(1)
plt.imshow(im_shift, label="shifted")
plt.show()

plt.figure(2)
plt.imshow(abs(im_rgb_ft) ,cmap='gray')
plt.show()

plt.figure(3)
plt.imshow(phase_rgb, label="magnitude")
plt.show()

rand=np.random.uniform(0, 1, (height, width))

rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
im_r_shift=exp_rand*im_shift


im_r_ft=fftshift(fft2(im_r_shift))
phase_r = np.angle(im_r_ft)

phase_r_new=phase_r.astype(np.uint8)
phase_r_save=Image.fromarray(phase_r_new)
# phase_r_save.save(f"{filename}_m_rand.png")

plt.figure(4)
plt.imshow(abs(im_r_shift), label="shifted")
plt.show()

plt.figure(5)
plt.imshow(abs(im_r_ft), label="magnitude")
plt.show()

plt.figure(6)
plt.imshow(phase_r, cmap="gray")
plt.show()
plt.imsave(f"{filename}_m_rand.png", phase_r,cmap='gray')