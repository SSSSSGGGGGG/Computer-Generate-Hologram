# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:45:57 2024

@author: gaosh
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import os
import cv2
import time


start_t=time.time()

os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6")
filename="One circle_1024"  #flowers_960 RGB_1024
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]

#R
im_shift_r=fftshift(im[:,:,0])

# random
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
#R
im_r_rand=exp_rand*np.sqrt(im_shift_r)

iterations1=40
iterations2=40

def n_win(current_field_r,iterations1):
    for j in range(iterations1):
        
        # Inverse Fourier Transform to initial plane
        current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
  
        current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))#*exp_rand
        
        # Forward Fourier Transform to the target plane
        current_field_r = fftshift(fft2(current_field_r_n ))
        return current_field_r 
        
l=300 # from edge to center 250 for 3circles 390
c_w,c_h=width//2,height//2
lh,lw=height-2*l,width-2*l
def win(current_field_r,iterations2):
    for i in range(iterations2):
        
        # Inverse Fourier Transform to initial plane
        current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
     
        current_field_r_i_t =np.sqrt(abs(current_field_r_i)**2/np.max(abs(current_field_r_i)**2)) *np.exp(1j * np.angle(current_field_r_i))
        
        current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))#*exp_rand
        
        current_field_r_n[:,c_w-l:c_w+l]=current_field_r_i_t[:,c_w-l:c_w+l]
        
        current_field_r_n[c_h-l:c_h+l,0:c_w-l]=current_field_r_i_t[c_h-l:c_h+l,0:c_w-l]
        
        current_field_r_n[c_h-l:c_h+l,c_w+l:width]=current_field_r_i_t[c_h-l:c_h+l,c_w+l:width]
        
        # Forward Fourier Transform to the target plane
        current_field_r = fftshift(fft2(current_field_r_n ))
        return current_field_r 

current_field =fftshift(fft2(im_r_rand))
        
FT_field_nw=n_win(current_field, iterations1)
FT_field_w=win(current_field, iterations2)

phase_nw=np.angle(FT_field_nw)
phase_w=np.angle(FT_field_w)

iFT_nw=ifftshift(ifft2(ifftshift(np.exp(1j*phase_nw))))
iFT_w=ifftshift(ifft2(ifftshift(np.exp(1j*phase_w))))
I_nw=abs(iFT_nw)**2
I_w=abs(iFT_w)**2

I_nw_n=I_nw/np.sum(I_nw)
I_w_n=I_w/np.sum(I_w) 
print(f"sum_nw={np.sum(I_nw)}, sum_w={np.sum(I_w)}")

wL=int((height-2*l)//2)
D_nw=(im[:,:,0]/np.sum(im[:,:,0])-I_nw_n)[c_h-wL:c_h+wL,c_w-wL:c_w+wL]
D_w=(im[:,:,0]/np.sum(im[:,:,0])-I_w_n)[c_h-wL:c_h+wL,c_w-wL:c_w+wL]

D2_nw=np.sqrt(np.sum((D_nw)**2))
D2_w=np.sqrt(np.sum((D_w)**2))
print(f"diff_nw={D2_nw}, diff_w={D2_w}")

plt.figure()
plt.imshow(abs(FT_field_nw),cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(abs(FT_field_w),cmap="hot")
plt.colorbar()
plt.show()

# plt.figure()
# plt.imshow(phase_nw,cmap="seismic")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(phase_w,cmap="seismic")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(D2_nw,cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(D2_w,cmap="hot")
# plt.colorbar()
# plt.show()
end_t=time.time()
print(f"Time consuming {end_t-start_t}s, iteration {iterations1+iterations2}")