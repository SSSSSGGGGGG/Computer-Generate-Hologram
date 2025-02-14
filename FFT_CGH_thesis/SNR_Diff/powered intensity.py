# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:07:05 2025

@author: ShangGao

"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, fftshift,ifft2,ifftshift

os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff")
filename="fl_one_s"  
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]
l=250
ofs=0
c_w=width//2+ofs
c_h=height//2+ofs
def powerf(p):
    
    power=p
    rand = np.random.uniform(0, 1, (height , width))
    rand_2pi = 2 * np.pi * rand
    p_im_r=np.sqrt(im[:,:,0]**power)#* np.exp(1j * rand_2pi)
    p_im_g=np.sqrt(im[:,:,1]**power)#* np.exp(1j * rand_2pi) 
    p_im_b=np.sqrt(im[:,:,2]**power)#* np.exp(1j * rand_2pi)
    
    
    print(f"power={p}: max {np.max(abs(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l]))}-min {abs(np.min(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l]))}={abs(np.max(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l])-np.min(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l]))}")
    
    FT_r=fftshift(fft2(fftshift(p_im_r)))
    FT_g=fftshift(fft2(fftshift(p_im_g)))
    FT_b=fftshift(fft2(fftshift(p_im_b)))
    
    # FT_r=fftshift(fft2(p_im_r))
    # FT_g=fftshift(fft2(p_im_g))
    # FT_b=fftshift(fft2(p_im_b))
    
    FT_r_m=np.abs(FT_r)
    FT_g_m=np.abs(FT_g)
    FT_b_m=np.abs(FT_b)
    
    return np.abs(p_im_r),FT_r_m, np.angle(FT_r)

P1=powerf(1)
P2=powerf(2)
P3=powerf(3)
P4=powerf(4)

# plt.figure()
# plt.imshow(P1[0],cmap="Reds")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P2[0],cmap="Reds")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P3[0],cmap="Reds")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P4[0],cmap="Reds")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P1[1][c_h-l:c_h+l,c_w-l:c_w+l]/np.max(P1[1]),cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P2[1][c_h-l:c_h+l,c_w-l:c_w+l]/np.max(P2[1]),cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P3[1][c_h-l:c_h+l,c_w-l:c_w+l]/np.max(P3[1]),cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(P4[1][c_h-l:c_h+l,c_w-l:c_w+l]/np.max(P4[1]),cmap="hot")
# plt.colorbar()
# plt.show()

crop_1=P1[2][c_h-l:c_h+l,c_w-l:c_w+l]
crop_2=P2[2][c_h-l:c_h+l,c_w-l:c_w+l]
crop_3=P3[2][c_h-l:c_h+l,c_w-l:c_w+l]
crop_4=P4[2][c_h-l:c_h+l,c_w-l:c_w+l]
# Inverse
# iF1=ifft2(ifftshift(np.exp(1j*crop_1)))
# iF2=ifft2(ifftshift(np.exp(1j*crop_2)))
# iF3=ifft2(ifftshift(np.exp(1j*crop_3)))
# iF4=ifft2(ifftshift(np.exp(1j*crop_4)))

# iF1=ifftshift(ifft2(ifftshift(np.exp(1j*crop_1))))
# iF2=ifftshift(ifft2(ifftshift(np.exp(1j*crop_2))))
# iF3=ifftshift(ifft2(ifftshift(np.exp(1j*crop_3))))
# iF4=ifftshift(ifft2(ifftshift(np.exp(1j*crop_4))))

iF1=ifftshift(ifft2(np.exp(1j*crop_1)))
iF2=ifftshift(ifft2(np.exp(1j*crop_2)))
iF3=ifftshift(ifft2(np.exp(1j*crop_3)))
iF4=ifftshift(ifft2(np.exp(1j*crop_4)))

# plt.figure()
# plt.imshow(crop_1,cmap="hot")#/np.max(P1[1])
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(crop_2,cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(crop_3,cmap="hot")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(crop_4,cmap="hot")
# plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(abs(iF1),cmap="Reds")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(abs(iF2),cmap="Reds")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(abs(iF3),cmap="Reds")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(abs(iF4),cmap="Reds")
plt.colorbar()
plt.show()

h=P1[1].shape[0]
w=P1[1].shape[1]
c_hs=h//2
c_ws=w//2
max_=np.max(P1[1][c_hs, :])
le=10
plt.figure()
plt.plot(P1[1][c_hs,c_ws-le :c_ws+le]/np.max(P1[1]), label=f'p=1, center')
plt.plot(P2[1][c_hs, c_ws-le :c_ws+le]/np.max(P2[1]), label=f'p=2, center')
plt.plot(P3[1][c_hs, c_ws-le :c_ws+le]/np.max(P3[1]), label=f'p=3, center')
plt.plot(P4[1][c_hs, c_ws-le :c_ws+le]/np.max(P4[1]), label=f'p=4, center')
plt.xlabel('Column Index')
# plt.ylim(0,max_)
plt.ylabel('Magnitude Value')
plt.title('Magnitude Profile Along Row')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(P2[1][c_hs, :]/np.sum(P2[1][c_hs, :]), label=f'p=2,Row {c_hs}')
# plt.xlabel('Column Index')
# # plt.ylim(0,max_)
# plt.ylabel('Intensity Value')
# plt.title('Intensity Profile Along Row')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(P3[1][c_hs, :]/np.sum(P3[1][c_hs, :]), label=f'p=3,Row {c_hs}')
# plt.xlabel('Column Index')
# # plt.ylim(0,max_)
# plt.ylabel('Intensity Value')
# plt.title('Intensity Profile Along Row')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(P4[1][c_hs, :]/np.sum(P4[1][c_hs, :]), label=f'p=4, Row {c_hs}')
# plt.xlabel('Column Index')
# # plt.ylim(0,max_)
# plt.ylabel('Intensity Value')
# plt.title('Intensity Profile Along Row')
# plt.legend()
# plt.show()

