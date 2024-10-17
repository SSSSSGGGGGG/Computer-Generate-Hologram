# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""
from PIL import Image
import numpy as np
import scipy as sp
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import matplotlib.pyplot as plt
import os
from skimage import color

square=np.zeros((512,512))
width,height=square.shape
center=int(width/2)
width_c=10
square[center-width_c:center+width_c,center-width_c:center+width_c]=1
plt.imsave("square.png", square,cmap='gray')

square_fft=fftshift(fft2(square))
square_mag=abs(square_fft)/np.max(abs(square_fft))
# plt.imshow(rect_mag,cmap="hot")
# plt.colorbar()
plt.imsave("square_fft.png", square_mag,cmap='hot')

rect=np.zeros((512,512))
rect[center-2*width_c:center+2*width_c,center-width_c:center+width_c]=1
plt.imsave("rect.png", rect,cmap='gray')

rect_fft=fftshift(fft2(rect))
rect_fft_mag=abs(rect_fft)/np.max(abs(rect_fft))
plt.imsave("rect_fft.png", rect_fft_mag,cmap='hot')

slit=np.zeros((512,512))
slit[:,int(center)-int(1*width_c):int(center)+int(1*width_c)]=1
plt.imsave("bar.png", slit,cmap='gray')

slit_fft=fftshift(fft2(slit))
slit_mag=abs(slit_fft)/np.max(abs(slit_fft))
plt.imsave("bar_fft.png", slit_mag,cmap='hot')
# plt.imshow(bar_mag,cmap="hot")
# plt.colorbar()