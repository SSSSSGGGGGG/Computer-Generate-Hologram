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


<<<<<<< HEAD
os.chdir("C:/Users/Laboratorio/MakeHologram/OriginalImage")
filename="ele_half.png"
=======
os.chdir("C:/Users/gaosh/Documents/python/Digital-hologram/OriginalImage")
filename="1.jpg"
>>>>>>> 6ea46742f7ffac311c80328b5db84c55737af1cb
im=plt.imread(filename)[:,:,:3]
shift_state="shift" # to check the shift influence,

im_r=im[:,:,0]

im_g=im[:,:,1]

im_b=im[:,:,2]



im_r_process=(im_r-1)*(-1)*255
# im_r_fft=fftshift(fft2(im_r))
<<<<<<< HEAD
im_r_fft=fftshift(ifft2(im_r_process))
=======
im_r_fft=fftshift(fft2(im_r))
im_g_fft=fftshift(fft2(im_g))
im_b_fft=fftshift(fft2(im_b))
>>>>>>> 6ea46742f7ffac311c80328b5db84c55737af1cb

rows, cols = im_r.shape
# Frequency coordinates
u = np.fft.fftshift(np.fft.fftfreq(rows))  # Frequency coordinates along rows
v = np.fft.fftshift(np.fft.fftfreq(cols))  # Frequency coordinates along columns
# print(u*rows,v*cols)
# Create frequency grid
U, V = np.meshgrid(u, v)

#(-pi, pi]
phase = np.angle(im_r_fft)+np.pi
<<<<<<< HEAD
mag=np.abs(im_r_fft)
=======
phase_g = np.angle(im_g_fft)+np.pi
phase_b = np.angle(im_b_fft)+np.pi
>>>>>>> 6ea46742f7ffac311c80328b5db84c55737af1cb
# the calibration between red and gray level 1.85-->255
interval=255/(2*np.pi)
angle_0_pi = phase*interval

interval_g=200/(2*np.pi)
angle_0_pi_g = phase_g*interval_g

interval_b=160/(2*np.pi)
angle_0_pi_b = phase_b*interval_b

# angle_0_pi =angle_0_pi # this is the one in 0-255
plt.figure(3)
plt.imshow(phase,cmap="hot")
plt.savefig("holo.tif")
plt.show()

<<<<<<< HEAD
plt.figure(1)
plt.imshow(mag,cmap="hot")
plt.savefig("holo.tif")
plt.show()

im_new=np.zeros_like(im)
im_new[:,:,0]=angle_0_pi
=======

plt.figure(2)
# plt.imshow(np.log(1+abs(im_r_fft)),cmap="gray")
plt.imshow(phase_g,cmap="gray",extent=[v.min(), v.max(), u.min(), u.max()], origin='lower', aspect='auto')
plt.show()

im_new=np.zeros_like(im)
im_new[:,:,0]=np.mod(angle_0_pi,255)
im_new[:,:,1]=np.mod(angle_0_pi_g,255)
im_new[:,:,2]=np.mod(angle_0_pi_b,255)
>>>>>>> 6ea46742f7ffac311c80328b5db84c55737af1cb
im_new_array = im_new.astype(np.uint8)

im_new_t = Image.fromarray(im_new_array)
im_new_t.save(f"ft of ele{interval}.png")

<<<<<<< HEAD
im_saved=plt.imread(f"ft of ele{interval}.png")[:,:,0]*255
phase_2=im_saved/interval

# combined = np.exp(1j * phase)
im_r_if = fft2(np.exp(1j * phase_2))

# I_r=im_r_if*im_r_if.conjugate()
plt.figure(2)
plt.imshow(np.abs(im_r_if),cmap="hot")
=======

im_saved=plt.imread("C:/Users/gaosh/Documents/python/Digital-hologram/OriginalImage/ft_of_1.jpg_resized.png")*255
# phase_2=im_saved/interval
plt.figure(3)
plt.imshow(im_saved)
plt.show()

# combined = np.exp(1j * phase)
im_r_if = ifft2(fftshift(np.exp(1j * phase_g)))

# I_r=im_r_if*im_r_if.conjugate()
plt.figure(1)
plt.imshow(im_new,cmap="hot",extent=[v.min(), v.max(), u.min(), u.max()], origin='lower', aspect='auto')
>>>>>>> 6ea46742f7ffac311c80328b5db84c55737af1cb
plt.show()
plt.savefig(f"ift of {filename} {interval}.tif")
