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
os.chdir("C:/Users/gaosh/Documents/python/Computer-Generate-Hologram/Con_Cor")
filename="Lotus" 
im_o=plt.imread(f"{filename}.png")
h,w=im_o[:,:,0].shape

height=1920
width=1920

center_h=height//2
center_w=width//2
# Genaration of random phase noise.
rand=np.random.uniform(0, 1, (h, w,3))
rand_2pi=2*np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
exp_rand_conjugate=np.exp(-1j*rand_2pi)

im_noise=im_o[:,:,:3]*exp_rand
im_noise_conjugate=im_o[:,:,:3]*exp_rand_conjugate
# im_c = np.zeros((height, width, 3), dtype=complex)
# im_c[center_h-150:center_h+150,center_w-150:center_w+150]=im_noise

length=-400
im_v = np.zeros((height, width, 3), dtype=complex)
im_v[center_h-150+length:center_h+150+length,center_w-150:center_w+150]=im_noise

im_l = np.zeros((height, width, 3), dtype=complex)
im_l[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise

# im_l_con = np.zeros((height, width, 3), dtype=complex)
# im_l_con[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise_conjugate

im_noise_con_flipped = np.flip(im_noise_conjugate, axis=(0, 1))  # Flip both vertically & horizontally
im_l_fl = np.zeros((height, width, 3), dtype=complex)
im_l_fl[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise_con_flipped


# rdp,gdp,bdp=1.85,2.63,3.55
def FFT(im):  
    # R
    im_shift_r=fftshift(im[:,:,0])
    #G
    im_shift_g=fftshift(im[:,:,1])
    #B
    im_shift_b=fftshift(im[:,:,2])
    
    im_r_rand = np.abs(im_shift_r) ** 0.5 * np.exp(1j * np.angle(im_shift_r))
    
    im_g_rand = np.abs(im_shift_g) ** 0.5 * np.exp(1j * np.angle(im_shift_g))
    
    im_b_rand = np.abs(im_shift_b) ** 0.5 * np.exp(1j * np.angle(im_shift_b))
    # FFT of different channels
    # FFT of different channels
    current_field_r =fftshift(fft2(im_r_rand ))
    current_field_g =fftshift(fft2(im_g_rand))
    current_field_b =fftshift(fft2(im_b_rand))     
    # Convert phase to grayscale.
    optimized_phase_r = np.angle(current_field_r)   
    optimized_phase_g = np.angle(current_field_g)  
    optimized_phase_b = np.angle(current_field_b)
    
    # phase_r_conv=(optimized_phase_r/np.pi+1)*(255/rdp)
    # phase_g_conv=(optimized_phase_g/np.pi+1)*(255/gdp)
    # phase_b_conv=(optimized_phase_b/np.pi+1)*(255/bdp)
    
    return optimized_phase_r,optimized_phase_g,optimized_phase_b

v_phase_r,v_phase_g,v_phase_b=FFT(im_v)
h_phase_r,h_phase_g,h_phase_b=FFT(im_l)
# h_phase_r_con,h_phase_g_con,h_phase_b_con=FFT(im_l_con)
h_phase_r_fl,h_phase_g_fl,h_phase_b_fl=FFT(im_l_fl)

def con(phase_r1,phase_r2,phase_g1,phase_g2,phase_b1,phase_b2):
    con_r=np.exp(1j*phase_r1)*np.exp(1j*phase_r2) 
    con_r=np.angle(con_r)
    con_g=np.exp(1j*phase_g1)*np.exp(1j*phase_g2) 
    con_g=np.angle(con_g)
    con_b=np.exp(1j*phase_b1)*np.exp(1j*phase_b2) 
    con_b=np.angle(con_b)
    
    return con_r,con_g,con_b

def corr(phase_r1,phase_r2,phase_g1,phase_g2,phase_b1,phase_b2):
    cor_r=np.exp(1j*phase_r1)*np.exp(-1j*phase_r2) 
    cor_r=np.angle(cor_r)
    cor_g=np.exp(1j*phase_g1)*np.exp(-1j*phase_g2) 
    cor_g=np.angle(cor_g)
    cor_b=np.exp(1j*phase_b1)*np.exp(-1j*phase_b2) 
    cor_b=np.angle(cor_b)
    return cor_r,cor_g,cor_b

V_con_r,V_con_g,V_con_b=con(v_phase_r, h_phase_r, v_phase_g, h_phase_g, v_phase_b, h_phase_b)
V_cor_r,V_cor_g,V_cor_b=corr(v_phase_r, h_phase_r, v_phase_g, h_phase_g, v_phase_b, h_phase_b)

# VHim_con_r,VHim_con_g,VHim_con_b=con(v_phase_r, h_phase_r_con, v_phase_g, h_phase_g_con, v_phase_b, h_phase_b_con)
# VHim_cor_r,VHim_cor_g,VHim_cor_b=corr(v_phase_r, h_phase_r_con, v_phase_g, h_phase_g_con, v_phase_b, h_phase_b_con)

VHfl_con_r,VHfl_con_g,VHfl_con_b=con(v_phase_r, h_phase_r_fl, v_phase_g, h_phase_g_fl, v_phase_b, h_phase_b_fl)
VHfl_cor_r,VHfl_cor_g,VHfl_cor_b=corr(v_phase_r, h_phase_r_fl, v_phase_g, h_phase_g_fl, v_phase_b, h_phase_b_fl)
# plt.figure()
# plt.imshow(np.abs(ifftshift(ifft2(np.exp(1j*VHfl_con_r)))),cmap="hot")
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow(np.abs(ifftshift(ifft2(np.exp(1j*VHfl_cor_r)))),cmap="hot")
# plt.colorbar()
# plt.show()
def IFFT(p_r,p_g,p_b):
    IF_r=ifftshift(ifft2(np.exp(1j*p_r)))
    IF_g=ifftshift(ifft2(np.exp(1j*p_g)))
    IF_b=ifftshift(ifft2(np.exp(1j*p_b)))
    
    return abs(IF_r),abs(IF_g),abs(IF_b)

Vcon_ir,Vcon_ig,Vcon_ib=IFFT(V_con_r,V_con_g,V_con_b)
Vcor_ir,Vcor_ig,Vcor_ib=IFFT(V_cor_r,V_cor_g,V_cor_b)

# Vim_con_ir,Vim_con_ig,Vim_con_ib=IFFT(VHim_con_r,VHim_con_g,VHim_con_b)
# Vim_cor_ir,Vim_cor_ig,Vim_cor_ib=IFFT(VHim_cor_r,VHim_cor_g,VHim_cor_b)

Vfl_con_ir,Vfl_con_ig,Vfl_con_ib=IFFT(VHfl_con_r,VHfl_con_g,VHfl_con_b)
Vfl_cor_ir,Vfl_cor_ig,Vfl_cor_ib=IFFT(VHfl_cor_r,VHfl_cor_g,VHfl_cor_b)

def display(i_r,name):
    
    plt.figure()
    plt.imshow(i_r,cmap="hot")
    plt.colorbar()
    plt.axis("off")
    plt.show()
    indices = np.where(i_r == 1)  # Assuming you want the red channel
    x_index, y_index = indices[1], indices[0]
    print(f"Pixel with value 1 found at (x={x_index}, y={y_index})")

    # x = np.arange(0, width)  
    # y = np.arange(0, height) 
    # X,Y=np.meshgrid(x,y)   
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, i_r, cmap='jet')
    # ax.set_zlim(0, 0.000002) 
    # ax.view_init(elev=35, azim=-20) 
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # # plt.savefig(f"Convolution {file1} {t} 3d.png", dpi=300, bbox_inches="tight")
    # plt.show()
di_1=display(Vcon_ir,"Con V+H0")
di_1=display(Vcor_ir,"Cor V+H0")    

di_1=display(Vfl_con_ir,"Con V+H0_im fl")
di_1=display(Vfl_cor_ir,"Cor V+H0_im fl") 
    
end_t=time.time()
print(f"Time consuming {end_t-start_t}s")