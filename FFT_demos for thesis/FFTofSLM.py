# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

"""
- A square white aperture of 64*64 pixels is used at the centre of a 512*512 pixels image.
- Multiple slits along vertical and horizontal directions, separated evenly, to create an amplitude grating.
- A 2D version is calculated as well.
"""
square=np.zeros((512,512))
width,height=square.shape
center=int(width/2)
width_c=32
period=16
half_period=int(period/2)
square[center-width_c:center+width_c,center-width_c:center+width_c]=1
for i in range(center - width_c, center + width_c, period):  
    square[center - width_c:center + width_c, i:i + half_period] = 0  
plt.imsave(f"Vertical slits p_{period}.png", square,cmap='gray')
"""2D FFT is applied to the shape of multiple vertical slits. """
square_fft=fftshift(fft2(square))
square_mag=abs(square_fft)/np.max(abs(square_fft))
plt.imsave(f"FFT_Vslits p_{period}.png", square_mag,cmap='hot')
"""2D FFT is applied to the shape of multiple horizontal slits. """
square_h=np.zeros((512,512))
width_h,height_h=square_h.shape
center_h=int(width_h/2)
square_h[center_h-width_c:center_h+width_c,center_h-width_c:center_h+width_c]=1
for i in range(center_h-width_c, center_h+width_c, period):  
    square_h[i:i + half_period,center_h - width_c:center_h+width_c] = 0  
plt.imsave(f"Horizontal slits p_{period}.png", square_h,cmap='gray')        

square_h_fft=fftshift(fft2(square_h))
square_h_mag=abs(square_h_fft)/np.max(abs(square_h_fft))
plt.imsave(f"FFT_Hslits p_{period}.png", square_h_mag,cmap='hot')
"""2D FFT is applied to the shape of a grid resembling an SLM. """
SLM=np.zeros((512,512))
width_s,height_s=SLM.shape
center_s=int(width_s/2)
SLM[center_s-width_c:center_s+width_c,center_s-width_c:center_s+width_c]=1
for i in range(center_h-width_c, center_h+width_c, period):  
    SLM[i:i + half_period,center_s - width_c:center_s+width_c] = 0 
    SLM[center - width_c:center + width_c, i:i + half_period] = 0
plt.imsave(f"SLM p_{period}.png", SLM,cmap='gray')

SLM_fft=fftshift(fft2(SLM))
SLM_mag=abs(SLM_fft)/np.max(abs(SLM_fft))
plt.imsave(f"FFT_SLM p_{period}.png", SLM_mag,cmap='hot')
# plt.imshow(SLM_mag,cmap="hot")