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
import time


start_t=time.time()

os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6")
filename="One circle_1024"  
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]

randP=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*randP
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)

l=300  # Length that is from edge to center.
c_w,c_h=width//2,height//2 # Center
lh,lw=height-2*l,width-2*l  # Height and width for the window.
im_crop=im[:,:,0][c_h-int(lh//2):c_h+int(lh//2),c_w-int(lw//2):c_w+int(lw//2)]
#R channel
im_shift_r=fftshift(im[:,:,0])

im_r_rand=np.sqrt(im_shift_r)#*exp_rand
# FT of the field without the window.
nw_field_r =fftshift(fft2(im_r_rand))
nw_if_1=ifft2(ifftshift(np.exp(1j * np.angle(nw_field_r))))
nw_ft_2=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(nw_if_1))))
nw_if_2=ifft2(ifftshift(np.exp(1j * np.angle(nw_ft_2))))
nw_ft_3=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(nw_if_2))))
nw_if_3=ifft2(ifftshift(np.exp(1j * np.angle(nw_ft_3))))
nw_ft_4=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(nw_if_3))))
nw_if_4=ifft2(ifftshift(np.exp(1j * np.angle(nw_ft_4))))
nw_ft_5=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(nw_if_4))))
# Inverse FFT2 computation of phase that is for the field with the window.
nw_field_r_if = ifftshift(ifft2(ifftshift(np.exp(1j * np.angle(nw_ft_5)))))
I_nw=abs(nw_field_r_if)**2 # Intensity of no winodw case
I_nw_crop=I_nw[c_h-int(lh//2):c_h+int(lh//2),c_w-int(lw//2):c_w+int(lw//2)]
I_nw_crop_nr=I_nw_crop/np.sum(I_nw)
I_nw_crop_nr=I_nw_crop_nr/np.max(I_nw_crop_nr)
sum_nw=np.sum(I_nw_crop_nr)

r=194
sum_r_n=0
sum_BG_n=0
hn,wn=I_nw_crop.shape
mask = np.zeros_like(I_nw_crop_nr, dtype=bool)
for jn in range(hn):
    for kn in range(wn):
        if (kn-wn//2)**2+(jn-hn//2)**2<=r**2:
            sum_r_n+=I_nw_crop_nr[jn,kn]
            mask[jn, kn] = True
        else:
            sum_BG_n+=I_nw_crop_nr[jn,kn]
sum_r_n_nr=sum_r_n/np.sum(I_nw)
sum_BG_n_nr=sum_BG_n/np.sum(I_nw)

plt.figure()
plt.imshow(I_nw_crop_nr, cmap="hot")
plt.colorbar()
plt.show()


sum_r_w=[]
sum_win_w=[]
BG_I=[]

# Random amplitude noise generation.
n=np.arange(0,0.1,0.01)
for i, v in enumerate(n):
    rand=np.random.uniform(0, v, (height, width))  
    # In the window, the amplitudes are from original. Otherwise, the amplitudes are generated amplitude noise.
    im_r_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]
    im_r_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
    im_r_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
    # FT of the field with the window.
    w_field_r =fftshift(fft2(im_r_rand))
    w_if_1=ifft2(ifftshift(np.exp(1j * np.angle(w_field_r))))
    
    im_r_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]
    im_r_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
    im_r_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
    
    w_ft_2=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(w_if_1))))
    w_if_2=ifft2(ifftshift(np.exp(1j * np.angle(w_ft_2))))
    
    im_r_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l] 
    im_r_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
    im_r_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
    
    w_ft_3=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(w_if_2))))
    w_if_3=ifft2(ifftshift(np.exp(1j * np.angle(w_ft_3))))
    
    im_r_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]
    im_r_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
    im_r_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
    
    w_ft_4=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(w_if_3))))
    w_if_4=ifft2(ifftshift(np.exp(1j * np.angle(w_ft_4))))
    
    im_r_rand[:,c_w-l:c_w+l]=rand[:,c_w-l:c_w+l]
    im_r_rand[c_h-l:c_h+l,0:c_w-l]=rand[c_h-l:c_h+l,0:c_w-l]
    im_r_rand[c_h-l:c_h+l,c_w+l:width]=rand[c_h-l:c_h+l,c_w+l:width]
    
    w_ft_5=fftshift(fft2(im_r_rand*np.exp(1j * np.angle(w_if_4))))
    # Inverse FFT2 computation of phase that is for the field without the window.
    w_field_r_if = ifftshift(ifft2(ifftshift(np.exp(1j * np.angle(w_ft_5)))))
    I_w=abs(w_field_r_if)**2 # Intensity of winodw case
    I_w_crop=I_w[c_h-int(lh//2):c_h+int(lh//2),c_w-int(lw//2):c_w+int(lw//2)]
    I_w_crop_nr=I_w_crop/np.sum(I_w)
    I_w_crop_nr=I_w_crop_nr/np.max(I_w_crop_nr)
    
    sum_w=np.sum(I_w_crop_nr)
    sum_win_w.append(sum_w)
    sum_r=0
    sum_BG=0
    h,w=I_w_crop.shape
    mask = np.zeros_like(I_w_crop_nr, dtype=bool)
    for j in range(h):
        for k in range(w):
            if (k-w//2)**2+(j-h//2)**2<=r**2:
                sum_r+=I_w_crop_nr[j,k]
                mask[j, k] = True 
            else:
                sum_BG+=I_w_crop_nr[j,k]
    
    sum_r_w.append(sum_r)
    BG_I.append(sum_BG)
    
    sum_win_w_nr=sum_win_w/np.sum(I_w)
    sum_r_w_nr=sum_r_w/np.sum(I_w)
    BG_I_nr=BG_I/np.sum(I_w)
    
I_w_circle = np.where(mask, I_w_crop, 0)
# plt.figure()
# plt.imshow(I_w_circle, cmap="hot")
# plt.colorbar()
# plt.show()
Diff_nw=im_crop-I_nw_crop_nr
Diff_w=im_crop-I_w_crop_nr
print(f"Diff_nw={np.sqrt(np.sum(Diff_nw))}, Diff_w={np.sqrt(np.sum(Diff_w))}, sum_nmcp={np.sum(I_nw_crop_nr)},sum_nmcp={np.sum(I_w_crop_nr)}")
plt.figure()
plt.imshow(I_w_crop_nr, cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
# plt.plot(n, sum_win_w_nr, label="I in the window with amplitude noise")
plt.plot(n, sum_win_w, label="I in the window with amplitude noise")
plt.plot(n, sum_r_w_nr, label="I in the circle with amplitude noise ")
plt.axhline(y=sum_r_n_nr, color='r', linestyle='--', label="I in the circle without amplitude noise")
plt.axhline(y=sum_nw, color='b', linestyle='--', label="I in the window without amplitude noise")
plt.xlabel("Noise amplitude (n)")
plt.ylabel("Total Intensity")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.imshow(I_nw,vmax=1e-4, cmap="hot")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(I_w,vmax=1e-4, cmap="hot")
plt.colorbar()
plt.show()

# plt.figure()
# plt.imshow(abs(nw_field_r), cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(abs(w_field_r), cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(np.angle(nw_field_r), cmap="hsv")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(np.angle(w_field_r), cmap="hsv")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(fftshift(abs(nw_field_r_if)), cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(fftshift(abs(w_field_r_if)), cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(fftshift(abs(nw_field_r_if)**2), cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(fftshift(abs(w_field_r_if)**2), cmap="hot")
# plt.colorbar()
# plt.show()