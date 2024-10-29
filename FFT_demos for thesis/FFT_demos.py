# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

"""
We define a 2D array with the size of 512*512 pixels, which is filled with 0 as background (in black color). We draw a small square with the size of 20*20 pixels that is filled with a value of 1 (in white color) in the array's center. The 2D FFT algorithm is applied for all the shapes, and "fftshift" is applied as well for shifting the low frequency to the center in the frequency domain. Finally, the magnitudes of FFT for different shapes are saved.
"""

square=np.zeros((512,512))
width,height=square.shape
center=int(width/2)
width_c=10
square[center-width_c:center+width_c,center-width_c:center+width_c]=1
plt.imsave("square.png", square,cmap='gray')

square_fft=fftshift(fft2(square))
square_mag=abs(square_fft)/np.max(abs(square_fft))
plt.imsave("square_fft.png", square_mag,cmap='hot')

rect=np.zeros((512,512))
rect[center-2*width_c:center+2*width_c,center-width_c:center+width_c]=1
plt.imsave("rect.png", rect,cmap='gray')

rect_fft=fftshift(fft2(rect))
rect_fft_mag=abs(rect_fft)/np.max(abs(rect_fft))
plt.imsave("rect_fft.png", rect_fft_mag,cmap='hot')

slit=np.zeros((512,512))
slit[:,int(center)-int(1*width_c):int(center)+int(1*width_c)]=1
plt.imsave("slit.png", slit,cmap='gray')

slit_fft=fftshift(fft2(slit))
slit_mag=abs(slit_fft)/np.max(abs(slit_fft))
plt.imsave("slit_fft.png", slit_mag,cmap='hot')
