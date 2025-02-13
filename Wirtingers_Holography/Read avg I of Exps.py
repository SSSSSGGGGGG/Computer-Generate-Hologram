# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:28:28 2025

@author: Shang Gao
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# os.chdir("C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/SNR_Diff small RGB and fl_TF/RGBw560 exp results")
os.chdir("C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/GS_GSwin exp/small target exp/power")

filename1="n1_0 n2_1 p3_s"  
im1=plt.imread(f"{filename1}.JPG")
filename="n1_0 n2_40 p3_s"  
im=plt.imread(f"{filename}.JPG")

height=im.shape[0]
width=im.shape[1]

l=1600//2
ofs=0
c_w=width//2+ofs
c_h=height//2+ofs

Im_inr1=im1[:,:,0][c_h-l:c_h+l,c_w-l:c_w+l]
Im_ing1=im1[:,:,1][c_h-l:c_h+l,c_w-l:c_w+l]
Im_inb1=im1[:,:,2][c_h-l:c_h+l,c_w-l:c_w+l]

Im_inr=im[:,:,0][c_h-l:c_h+l,c_w-l:c_w+l]
Im_ing=im[:,:,1][c_h-l:c_h+l,c_w-l:c_w+l]
Im_inb=im[:,:,2][c_h-l:c_h+l,c_w-l:c_w+l]

a_im_r=np.average(Im_inr)
a_im_g=np.average(Im_ing)
a_im_b=np.average(Im_inb)
# def avgI(Im_in):
    
#     l=680//2
#     ofs=0
#     c_w=width//2+ofs
#     c_h=height//2+ofs
    
#     a_im_r=np.average(Im_in[:,:,0][c_h-l:c_h+l,c_w-l:c_w+l])
#     a_im_g=np.average(Im_in[:,:,1][c_h-l:c_h+l,c_w-l:c_w+l])
#     a_im_b=np.average(Im_in[:,:,2][c_h-l:c_h+l,c_w-l:c_w+l])
    
    
#     # print(f"power={p}: max {np.max(abs(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l]))}-min {abs(np.min(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l]))}={abs(np.max(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l])-np.min(p_im_r[c_h-l:c_h+l,c_w-l:c_w+l]))}")
    
    
#     return a_im_r,a_im_g,a_im_b

# R_avg,G_avg,B_avg=avgI(im)
# print(f"R {R_avg}, G {R_avg}, B {R_avg}")
plt.figure()
plt.imshow(Im_inr1,cmap="Reds")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Im_ing1,cmap="Greens")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Im_inb1,cmap="Blues")
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(Im_inr,cmap="Reds")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Im_ing,cmap="Greens")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Im_inb,cmap="Blues")
plt.colorbar()
plt.show()