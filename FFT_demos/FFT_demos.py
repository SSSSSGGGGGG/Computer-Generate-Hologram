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

rect=np.zeros((512,512))
width,height=rect.shape
center=int(width/2)
width_c=10
rect[center-width_c:center+width_c,center-width_c:center+width_c]=1
plt.imsave("square.png", rect,cmap='gray')

rect_fft=fftshift(fft2(rect))
rect_mag=abs(rect_fft)/np.max(abs(rect_fft))
# plt.imshow(rect_mag,cmap="hot")
# plt.colorbar()
plt.imsave("square_fft.png", rect_mag,cmap='hot')

rect_r=np.zeros((512,512))
rect_r[center-2*width_c:center+2*width_c,center-width_c:center+width_c]=1
plt.imsave("rect.png", rect_r,cmap='gray')

rect_r_fft=fftshift(fft2(rect_r))
rect_r_fft_mag=abs(rect_r_fft)/np.max(abs(rect_r_fft))
plt.imsave("rect_fft.png", rect_r_fft_mag,cmap='hot')

bar=np.zeros((512,512))
bar[:,int(center)-int(1*width_c):int(center)+int(1*width_c)]=1
plt.imsave("bar.png", bar,cmap='gray')

bar_fft=fftshift(fft2(bar))
bar_mag=abs(bar_fft)/np.max(abs(bar_fft))
plt.imsave("bar_fft.png", bar_mag,cmap='hot')
# plt.imshow(bar_mag,cmap="hot")
# plt.colorbar()