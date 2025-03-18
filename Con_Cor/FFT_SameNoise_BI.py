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
os.chdir("C:/Users/Laboratorio/MakeHologram/Con_Cor")
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

im_noise=im_o[:,:,:3]*exp_rand
# im_c = np.zeros((height, width, 3), dtype=complex)
# im_c[center_h-150:center_h+150,center_w-150:center_w+150]=im_noise

length=-400
im_v = np.zeros((height, width, 3), dtype=complex)
im_v[center_h-150+length:center_h+150+length,center_w-150:center_w+150]=im_noise

im_l = np.zeros((height, width, 3), dtype=complex)
im_l[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise

# im_noise_flipped = np.flip(im_noise, axis=(0, 1))  # Flip both vertically & horizontally
# im_l_fl = np.zeros((height, width, 3), dtype=complex)
# im_l_fl[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise_flipped

"""Rescaling is only applied for generating the hologams for experiment."""
# Define wavelengths (in meters, for rescale)
lambda_r = 0.662e-6  # Red wavelength
lambda_g = 0.518e-6  # Green wavelength (reference)
lambda_b = 0.449e-6  # Blue wavelength
# Calculate scaling factors with respect to green
scale_r =  lambda_r/lambda_g
scale_b =  lambda_b/lambda_g

h_r, w_r = int(height * scale_r), int(width * scale_r)
h_b, w_b = int(height * scale_b), int(width * scale_b)

x_offset_r = (int(w_r)-width) // 2
y_offset_r = (int(h_r)-height) // 2

x_offset_b = (width - int(w_b)) // 2
y_offset_b = (height - int(h_b)) // 2

def rescale(im):
    # Step 1: Pad the red channel to the scaled size
    padded_red = np.zeros((h_r, w_r), dtype=complex)
    padded_red[y_offset_r:y_offset_r + height, x_offset_r:x_offset_r + width] = im[:, :, 0]
    # Step 2: Crop the blue channel to the scaled size
    cropped_blue=np.zeros((h_b, w_b), dtype=complex)
    cropped_blue = im[:, :, 2][y_offset_b:y_offset_b + h_b, x_offset_b:x_offset_b + w_b]
    # Step 3: Resize both channels back to the original dimensions
    real_partr = cv2.resize(np.real(padded_red), (width, height), interpolation=cv2.INTER_LINEAR)
    imag_partr = cv2.resize(np.imag(padded_red), (width, height), interpolation=cv2.INTER_LINEAR)
    scaled_red = real_partr + 1j * imag_partr
    
    real_partb = cv2.resize(np.real(cropped_blue), (width, height), interpolation=cv2.INTER_LINEAR)
    imag_partb = cv2.resize(np.imag(cropped_blue), (width, height), interpolation=cv2.INTER_LINEAR)
    scaled_blue = real_partb + 1j * imag_partb  # Recombine into complex format
    #R
    im_shift_r=fftshift(scaled_red)
    #G
    im_shift_g=fftshift(im[:,:,1])
    #B
    im_shift_b=fftshift(scaled_blue)
    
    return im_shift_r,im_shift_g,im_shift_b
im_v_re_r,im_v_re_g,im_v_re_b=rescale(im_v)
im_l_re_r,im_l_re_g,im_l_re_b=rescale(im_l)
# im_l_re_r,im_l_re_g,im_l_re_b=rescale(im_l)

rdp,gdp,bdp=1.85,2.63,3.55
def FFT(im_shift_r,im_shift_g,im_shift_b):    
    
    im_r_rand = np.abs(im_shift_r) ** 0.5 * np.exp(1j * np.angle(im_shift_r))
    
    im_g_rand = np.abs(im_shift_g) ** 0.5 * np.exp(1j * np.angle(im_shift_g))
    
    im_b_rand = np.abs(im_shift_b) ** 0.5 * np.exp(1j * np.angle(im_shift_b))
    # FFT of different channels
    current_field_r =fftshift(fft2(im_r_rand ))
    current_field_g =fftshift(fft2(im_g_rand))
    current_field_b =fftshift(fft2(im_b_rand))     
    # Convert phase to grayscale.
    optimized_phase_r = np.angle(current_field_r)
    # phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/rdp)
    
    optimized_phase_g = np.angle(current_field_g)
    # phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/gdp)
    
    optimized_phase_b = np.angle(current_field_b)
    # phase_br_modi=(optimized_phase_b/np.pi+1)*(255/bdp)
    
    """Binarize"""
    
    optimized_phase_r_bi = np.where(optimized_phase_r < 0, 0, 1) # Now there is no complex field for binarized.
   
    optimized_phase_g_bi = np.where(optimized_phase_g < 0, 0, 1)
    
    optimized_phase_b_bi = np.where(optimized_phase_b < 0, 0, 1)
   
    return optimized_phase_r,optimized_phase_g,optimized_phase_b,optimized_phase_r_bi,optimized_phase_g_bi,optimized_phase_b_bi
v_phase_r,v_phase_g,v_phase_b,v_phase_r_bi,v_phase_g_bi,v_phase_b_bi=FFT(im_v_re_r,im_v_re_g,im_v_re_b)
h_phase_r,h_phase_g,h_phase_b,h_phase_r_bi,h_phase_g_bi,h_phase_b_bi=FFT(im_l_re_r,im_l_re_g,im_l_re_b)

def con_corr(phase_r1,phase_r2,phase_g1,phase_g2,phase_b1,phase_b2):
    con_r=phase_r1+phase_r2
    cor_r=phase_r1-phase_r2
    
    con_g=phase_g1+phase_g2
    cor_g=phase_g1-phase_g2
    
    con_b=phase_b1+phase_b2
    cor_b=phase_b1-phase_b2
    return con_r,cor_r,con_g,cor_g,con_b,cor_b

con_r,cor_r,con_g,cor_g,con_b,cor_b=con_corr(v_phase_r, h_phase_r, v_phase_g, h_phase_g, v_phase_b, h_phase_b)
# R with white
R_con_r,R_cor_r,R_con_g,R_cor_g,R_con_b,R_cor_b=con_corr(v_phase_r, h_phase_r, v_phase_g, 0, v_phase_b, 0)
# G with white
G_con_r,G_cor_r,G_con_g,G_cor_g,G_con_b,G_cor_b=con_corr(v_phase_r, 0, v_phase_g, h_phase_g, v_phase_b, 0)
# B with white
B_con_r,B_cor_r,B_con_g,B_cor_g,B_con_b,B_cor_b=con_corr(v_phase_r, 0, v_phase_g, 0, v_phase_b, h_phase_b)

def conv_gray(p_r,p_g,p_b,factor):
    # if the phase is single retrieved, factor 2 should not be here.
    nor_r=p_r/np.pi/factor
    nor_g=p_g/np.pi/factor
    nor_b=p_b/np.pi/factor
    phase_r_conv=(nor_r+1)*(255/rdp)
    phase_g_conv=(nor_g+1)*(255/gdp)
    phase_b_conv=(nor_b+1)*(255/bdp)
    
    return phase_r_conv,phase_g_conv,phase_b_conv

p_r_con,p_g_con,p_b_con=conv_gray(con_r, con_g, con_b,2)
p_r_cor,p_g_cor,p_b_cor=conv_gray(cor_r, cor_g, con_b,2)
# R with white
R_p_r_con,R_p_g_con,R_p_b_con=conv_gray(R_con_r, R_con_g, R_con_b,2)
R_p_r_cor,R_p_g_cor,R_p_b_cor=conv_gray(R_cor_r, R_cor_g, R_con_b,2)
# G with white
G_p_r_con,G_p_g_con,G_p_b_con=conv_gray(G_con_r, G_con_g, G_con_b,2)
G_p_r_cor,G_p_g_cor,G_p_b_cor=conv_gray(G_cor_r, G_cor_g, G_con_b,2)
# B with white
B_p_r_con,B_p_g_con,B_p_b_con=conv_gray(B_con_r, B_con_g, B_con_b,2)
B_p_r_cor,B_p_g_cor,B_p_b_cor=conv_gray(B_cor_r, B_cor_g, B_con_b,2)
# for single holograms
S_p_r_v,S_p_g_v,S_p_b_v=conv_gray(v_phase_r,v_phase_g,v_phase_b,1)
S_p_r_h,S_p_g_h,S_p_b_h=conv_gray(h_phase_r,h_phase_g,h_phase_b,1)


def con_cor_Bi(phase_r1,phase_r2,phase_g1,phase_g2,phase_b1,phase_b2,type_):
    if type_=="tri":
            gr,gg,gb=185,185,185
    else:
            type_="pi"
            gr,gg,gb=136,136,136
    con_r=phase_r1+phase_r2
    cor_r=phase_r1-phase_r2
    con_r_gray=con_r*gr
    cor_r_gray=cor_r*gr
    
    con_g=phase_g1+phase_g2
    cor_g=phase_g1-phase_g2
    con_g_gray=con_g*gg
    cor_g_gray=cor_g*gg
    
    con_b=phase_b1+phase_b2
    cor_b=phase_b1-phase_b2
    con_b_gray=con_b*gb
    cor_b_gray=cor_b*gb
    
    return con_r_gray,con_g_gray,con_b_gray,cor_r_gray,cor_g_gray,cor_b_gray
t="pi"
Bi_p_r_con,Bi_p_g_con,Bi_p_b_con,Bi_p_r_cor,Bi_p_g_cor,Bi_p_b_cor=con_cor_Bi(v_phase_r_bi, h_phase_r_bi, v_phase_g_bi, h_phase_g_bi, v_phase_b_bi, v_phase_b_bi, t) 
Bi_p_r_Sv,Bi_p_g_Sv,Bi_p_b_Sv=con_cor_Bi(v_phase_r_bi, 0, v_phase_g_bi, 0, v_phase_b_bi, 0, t)[:3]   
Bi_p_r_Sh,Bi_p_g_Sh,Bi_p_b_Sh=con_cor_Bi(0, h_phase_r_bi, 0, h_phase_g_bi, 0, h_phase_b_bi, t)[:3]   
t2="tri"
Bi_p_r_con2,Bi_p_g_con2,Bi_p_b_con2,Bi_p_r_cor2,Bi_p_g_cor2,Bi_p_b_cor2=con_cor_Bi(v_phase_r_bi, h_phase_r_bi, v_phase_g_bi, h_phase_g_bi, v_phase_b_bi, v_phase_b_bi, t)
Bi_p_r_Sv2,Bi_p_g_Sv2,Bi_p_b_Sv2=con_cor_Bi(v_phase_r_bi, 0, v_phase_g_bi, 0, v_phase_b_bi, 0, t2)[:3]   
Bi_p_r_Sh2,Bi_p_g_Sh2,Bi_p_b_Sh2=con_cor_Bi(0, h_phase_r_bi, 0, h_phase_g_bi, 0, h_phase_b_bi, t2)[:3]  
"""Lens"""
lambda_r = 0.662e-6  
lambda_g = 0.518e-6  
lambda_b = 0.449e-6  
arr_r=np.zeros((height, width))
arr_g=np.zeros((height, width))
arr_b=np.zeros((height, width))
pixel_size=4.5e-6
f=-2 # focal length (meters).
# Calculate the phase of lenses.
for i in range(height):
    for j in range(width):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        arr_r[i, j] =  r**2 / (f * lambda_r) 
        arr_g[i, j] =  r**2 / (f * lambda_g)
        arr_b[i, j] =  r**2 / (f * lambda_b)
# Change the range of the phase into [0,2].
arr_r_mod=np.mod(arr_r,2)
arr_g_mod=np.mod(arr_g,2)
arr_b_mod=np.mod(arr_b,2)
# Map the phase to grayscale by corresponding phase modulation.
arr_r_modified=arr_r_mod*(255/rdp)
arr_g_modified=arr_g_mod*(255/gdp)
arr_b_modified=arr_b_mod*(255/bdp)

def crop(im_modify,name):
    y_offset=center_h-1080//2
    im_cropped=im_modify[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{name}.png")
def saveim(p_r,p_g,p_b):
    # Save and crop the corresponding RGB CGHs with/without lens for experiment and simulation, respectively.
    im_modify_noL = np.zeros((height, width, 3), dtype=np.float32)
    im_modify_noL[:,:,0] = p_r
    im_modify_noL[:,:,1] = p_g
    im_modify_noL[:,:,2] = p_b
    
    im_modify_c = np.zeros((height, width, 3), dtype=np.float32)
    im_modify_c[:,:,0] = p_r+arr_r_modified
    im_modify_c[:,:,1] = p_g+arr_g_modified
    im_modify_c[:,:,2] = p_b+arr_b_modified
    
    im_modify_c_r = np.zeros((height, width, 3), dtype=np.float32)
    im_modify_c_r[:,:,0] = p_r+arr_r_modified
    im_modify_c_g = np.zeros((height, width, 3), dtype=np.float32)
    im_modify_c_g[:,:,1] = p_g+arr_g_modified
    im_modify_c_b = np.zeros((height, width, 3), dtype=np.float32)
    im_modify_c_b[:,:,2] = p_b+arr_b_modified
    
    return im_modify_c,im_modify_c_r,im_modify_c_g,im_modify_c_b
    
saved_1,saved_2,saved_3,saved_4=saveim(p_r_con,p_g_con,p_b_con)
im_l_save=crop(saved_1,f"VH0 con L")
saved_1,saved_2,saved_3,saved_4=saveim(R_p_r_con,R_p_g_con,R_p_b_con)
im_l_save=crop(saved_1,f"VH0 con R_L")
saved_1,saved_2,saved_3,saved_4=saveim(G_p_r_con,G_p_g_con,G_p_b_con)
im_l_save=crop(saved_1,f"VH0 con G_L")
saved_1,saved_2,saved_3,saved_4=saveim(B_p_r_con,B_p_g_con,B_p_b_con)
im_l_save=crop(saved_1,f"VH0 con B_L")

saved_1,saved_2,saved_3,saved_4=saveim(p_r_cor,p_g_cor,p_b_cor)
im_l_save=crop(saved_1,f"VH0 cor L")
saved_1,saved_2,saved_3,saved_4=saveim(R_p_r_cor,R_p_g_cor,R_p_b_cor)
im_l_save=crop(saved_1,f"VH0 cor R_L")
saved_1,saved_2,saved_3,saved_4=saveim(G_p_r_cor,G_p_g_cor,G_p_b_cor)
im_l_save=crop(saved_1,f"VH0 cor G_L")
saved_1,saved_2,saved_3,saved_4=saveim(B_p_r_cor,B_p_g_cor,B_p_b_cor)
im_l_save=crop(saved_1,f"VH0 cor B_L")

saved_1,saved_2,saved_3,saved_4=saveim(S_p_r_v,S_p_g_v,S_p_b_v)
im_l_save=crop(saved_1,f"VH0 Sv L")
im_lr_save=crop(saved_2,f"VH0 Sv R_L")
im_lg_save=crop(saved_3,f"VH0 Sv G_L")
im_lb_save=crop(saved_4,f"VH0 Sv B_L")
saved_1,saved_2,saved_3,saved_4=saveim(S_p_r_h,S_p_g_h,S_p_b_h)
im_l_save=crop(saved_1,f"VH0 Sh L")
im_lr_save=crop(saved_2,f"VH0 Sh R_L")
im_lg_save=crop(saved_3,f"VH0 Sh G_L")
im_lb_save=crop(saved_4,f"VH0 Sh B_L")

# saved_=saveim(Bi_p_r_con,Bi_p_g_con,Bi_p_b_con, f"Bi_{t} con")
# saved_=saveim(Bi_p_r_cor,Bi_p_g_cor,Bi_p_b_cor, f"Bi_{t} cor")
# saved_=saveim(Bi_p_r_Sv,Bi_p_g_Sv,Bi_p_b_Sv, f"Bi_{t} Sv")
# saved_=saveim(Bi_p_r_Sh,Bi_p_g_Sh,Bi_p_b_Sh, f"Bi_{t} Sh")

# saved_=saveim(Bi_p_r_con2,Bi_p_g_con2,Bi_p_b_con2, f"Bi_{t2} con")
# saved_=saveim(Bi_p_r_cor2,Bi_p_g_cor2,Bi_p_b_cor2, f"Bi_{t2} cor")
# saved_=saveim(Bi_p_r_Sv2,Bi_p_g_Sv2,Bi_p_b_Sv2, f"Bi_{t2} Sv")
# saved_=saveim(Bi_p_r_Sh2,Bi_p_g_Sh2,Bi_p_b_Sh2, f"Bi_{t2} Sh")

end_t=time.time()
print(f"Time consuming {end_t-start_t}s")