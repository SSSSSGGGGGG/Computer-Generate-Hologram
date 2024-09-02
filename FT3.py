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
filename="1"
im=plt.imread(f"{filename}.jpg")
holo_ex=plt.imread("Hologram (2).tiff")
# holo_ex_r=holo_ex[:,:,:3]

im_rgb_ft=fftshift(fft2(im))
phase_rgb = np.angle(im_rgb_ft)

plt.figure(1)
plt.imshow(phase_rgb)
plt.title("FT of phase")
plt.show()

im_new_array = phase_rgb.astype(np.uint8)
im_new_t = Image.fromarray(im_new_array)
# im_new_t.show()
im_new_t.save(f"{filename}_RGBholo.png")

im_rgb_ift=ifft2(np.exp(1j*phase_rgb))

plt.figure(2)
plt.imshow(abs(im_rgb_ift))
plt.title("iFT")
plt.show()
# # the following is for red chennel holo
# im_r=im[:,:,0]
# im_r_ft=fftshift(fft2(im_r))
# phase_r=np.angle(im_r_ft)

# plt.figure(3)
# plt.imshow(phase_r)
# plt.title("FT of R chennel phase")
# plt.show()

# im_new_r=np.zeros_like(im)
# im_new_r[:,:,0] = phase_r
# im_new_r = Image.fromarray(im_new_r.astype(np.uint8))
# im_new_r.save(f"{filename}_Rholo.png")

# im_new_gray=np.zeros_like(im)
# im_new_gray[:,:,0] = phase_r
# im_new_gray[:,:,1] = phase_r
# im_new_gray[:,:,2] = phase_r
# im_new_gray = Image.fromarray(im_new_gray.astype(np.uint8))
# im_rgb_gray=im_new_gray.convert('L')
# im_rgb_gray.save(f"{filename}_grayholo.png")

# plt.figure(4)
# plt.imshow(im_new_gray)
# plt.title("FT of gray phase")
# plt.show()