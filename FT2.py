# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image
import numpy as np
import scipy as sp
from scipy.fft import fft2, fftshift,ifft2
import matplotlib.pyplot as plt
import os
from skimage import color


os.chdir("C:/Users/Laboratorio/MakeHologram/OriginalImage")
im=plt.imread("1.jpg")


im_r=im[:,:,0]

im_g=im[:,:,1]

im_b=im[:,:,2]

im_gray=color.rgb2gray(im) # need to check


im_r_fft=fft2(im_r)
I_r=im_r_fft**2
im_r_if=ifft2(im_r_fft)
# plt.imshow(abs(I))
# plt.axis('off')
im_new=np.zeros_like(im)
im_new[:,:,0]=im_r_fft
im_new_array = im_new.astype(np.uint8)

im_new_t = Image.fromarray(im_new_array)
im_new_t.save("FT of car.png")
im_new_t.show()