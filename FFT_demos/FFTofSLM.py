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
width_c=30
square[center-width_c:center+width_c,center-width_c:center+width_c]=1
for i in range(center - width_c, center + width_c, 12):  # Step of 12 within the block
    # print(i)
    square[center - width_c:center + width_c, i:i + 6] = 0  # Apply to the defined region only

# plt.imsave("slits in square.png", square,cmap='gray')

square_fft=fftshift(fft2(square))
square_mag=abs(square_fft)/np.max(abs(square_fft))
# plt.imsave("slits in square_fft.png", square_mag,cmap='hot')
# plt.imshow(square_mag,cmap="hot")
# plt.colorbar()
square_h=np.zeros((512,512))
width_h,height_h=square_h.shape
center_h=int(width_h/2)
# width_c=30
square_h[center_h-width_c:center_h+width_c,center_h-width_c:center_h+width_c]=1
for i in range(center_h-width_c, center_h+width_c, 12):  # Step of 12 within the block
    # print(i)
    square_h[i:i + 6,center_h - width_c:center_h+width_c] = 0  # Apply to the defined region only
# plt.imsave("slits_h in square.png", square_h,cmap='gray')        

square_h_fft=fftshift(fft2(square_h))
square_h_mag=abs(square_h_fft)/np.max(abs(square_h_fft))
# plt.imsave("slits_h in square_fft.png", square_h_mag,cmap='hot')


SLM=np.zeros((512,512))
width_s,height_s=SLM.shape
center_s=int(width_s/2)
SLM[center_s-width_c:center_s+width_c,center_s-width_c:center_s+width_c]=1
for i in range(center_h-width_c, center_h+width_c, 12):  # Step of 12 within the block
    # print(i)
    SLM[i:i + 6,center_s - width_c:center_s+width_c] = 0 
    SLM[center - width_c:center + width_c, i:i + 6] = 0
# plt.imsave("SLM.png", SLM,cmap='gray')

SLM_fft=fftshift(fft2(SLM))
SLM_mag=abs(SLM_fft)/np.max(abs(SLM_fft))
plt.imsave("SLM_fft.png", SLM_mag,cmap='hot')