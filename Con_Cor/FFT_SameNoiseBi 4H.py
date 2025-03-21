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
exp_rand_conjugate=np.exp(-1j*rand_2pi)


im_noise=im_o[:,:,:3]*exp_rand
im_noise_conjugate=im_o[:,:,:3]*exp_rand_conjugate
im_noise_con_flipped = np.flip(im_noise_conjugate, axis=(0, 1))  # Flip both vertically & horizontally

im_c = np.zeros((height, width, 3), dtype=complex)
im_c[center_h-150:center_h+150,center_w-150:center_w+150]=im_noise

im_c_fl = np.zeros((height, width, 3), dtype=complex)
im_c_fl[center_h-150:center_h+150,center_w-150:center_w+150]=im_noise_con_flipped

# plt.figure()
# plt.imshow(abs(im_c))
# plt.figure()
# plt.imshow(abs(im_c_fl))

length=-400
im_v = np.zeros((height, width, 3), dtype=complex)
im_v[center_h-150+length:center_h+150+length,center_w-150:center_w+150]=im_noise

im_l = np.zeros((height, width, 3), dtype=complex)
im_l[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise
# plt.imsave("original center L.png", abs(im_l)**2, cmap="gray")
im_l_con = np.zeros((height, width, 3), dtype=complex)
im_l_con[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise_conjugate


im_l_fl = np.zeros((height, width, 3), dtype=complex)
im_l_fl[center_h-150:center_h+150,center_w-150+length:center_w+150+length]=im_noise_con_flipped


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
print(f"y_offset_r: {y_offset_r}, x_offset_r: {x_offset_r}")
print(f"Padded size: {h_r}x{w_r}, Image size: {height}x{width}")

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
# im_v_re_r,im_v_re_g,im_v_re_b=rescale(im_v)
# im_l_re_r,im_l_re_g,im_l_re_b=rescale(im_l)
im_l_re_r_fl,im_l_re_g_fl,im_l_re_b_fl=rescale(im_l_fl)

im_l_re_r,im_l_re_g,im_l_re_b=rescale(im_c)
# im_l_re_r_fl,im_l_re_g_fl,im_l_re_b_fl=rescale(im_c_fl)

rdp,gdp,bdp=1.85,2.63,3.55
def FFT_nB(im_shift_r,im_shift_g,im_shift_b):    
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
    
    return optimized_phase_r,optimized_phase_g,optimized_phase_b

def convToGray_nB(phase_r,phase_g,phase_b):
    
    r_gray=(phase_r/np.pi+1)*(255/rdp)
    
    g_gray=(phase_g/np.pi+1)*(255/gdp)
    
    b_gray=(phase_b/np.pi+1)*(255/bdp)
    
    return r_gray,g_gray,b_gray

def FFT(im_shift_r,im_shift_g,im_shift_b):    
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
    
    optimized_phase_r_bi = np.where(optimized_phase_r < 0, 0, 1) # Now there is no complex field for binarized.
   
    optimized_phase_g_bi = np.where(optimized_phase_g < 0, 0, 1)
    
    optimized_phase_b_bi = np.where(optimized_phase_b < 0, 0, 1)
    return optimized_phase_r_bi,optimized_phase_g_bi,optimized_phase_b_bi

# v_phase_r,v_phase_g,v_phase_b=FFT(im_v_re_r,im_v_re_g,im_v_re_b)
h_phase_r,h_phase_g,h_phase_b=FFT_nB(im_l_re_r,im_l_re_g,im_l_re_b)

h_phase_r_fl,h_phase_g_fl,h_phase_b_fl=FFT(im_l_re_r_fl,im_l_re_g_fl,im_l_re_b_fl)

def convToGray(phase_r,phase_g,phase_b,type_):
    if type_=="tri":
            gr,gg,gb=185,185,185
    else:
            gr,gg,gb=136,136,136
    
    r_gray=phase_r*gr
    
    g_gray=phase_g*gg
    
    b_gray=phase_b*gb
    
    return r_gray,g_gray,b_gray

def con_cor_Bi(phase_r1,phase_r2,phase_g1,phase_g2,phase_b1,phase_b2,type_):
    if type_=="tri":
            gr,gg,gb=185,185,185
    else:
            gr,gg,gb=136,136,136
    phase_r2_gray=phase_r2*0.64*np.pi
    phase_g2_gray=phase_g2*0.64*np.pi
    phase_b2_gray=phase_b2*0.64*np.pi
    
    phase_r1_gray=phase_r1+np.pi
    phase_g1_gray=phase_g1+np.pi
    phase_b1_gray=phase_b1+np.pi
    
    
    con_r=np.exp(1j*phase_r1)*np.exp(1j*phase_r2) 
    con_r=np.angle(con_r)+np.pi
    con_r=np.mod(con_r,2*np.pi)
    
    con_g=np.exp(1j*phase_g1)*np.exp(1j*phase_g2) 
    con_g=np.angle(con_g)+np.pi
    con_g=np.mod(con_g,2*np.pi)
    
    con_b=np.exp(1j*phase_b1)*np.exp(1j*phase_b2) 
    con_b=np.angle(con_b)+np.pi
    con_b=np.mod(con_b,2*np.pi)       
    
    # con_r_gray=con_r*gr
    # con_g_gray=con_g*gg
    # con_b_gray=con_b*gb
    con_r_gray,con_g_gray,con_b_gray=convToGray_nB(con_r, con_g, con_b)
    
    return con_r_gray,con_g_gray,con_b_gray

t="tri"
# con_r,con_g,con_b=con_cor_Bi(v_phase_r, h_phase_r, v_phase_g, h_phase_g, v_phase_b, h_phase_b,t)
# cor_r,cor_g,cor_b=con_cor_Bi(v_phase_r, h_phase_r_fl, v_phase_g, h_phase_g_fl, v_phase_b, h_phase_b_fl,t)

# con_r,con_g,con_b=con_cor_Bi(h_phase_r, h_phase_r, h_phase_g, h_phase_g, h_phase_b, h_phase_b,t)
cor_r,cor_g,cor_b=con_cor_Bi(h_phase_r, h_phase_r_fl, h_phase_g, h_phase_g_fl, h_phase_b, h_phase_b_fl,t)

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

# saved_1=saveim(con_r,con_g,con_b)[0]
# im_l_save=crop(saved_1,f"Bi_VH0 {t} con L")

# saved_1=saveim(cor_r,cor_g,cor_b)[0]
# im_l_save=crop(saved_1,f"Bi_VH180 {t} cor L")

# saved_1=saveim(con_r,con_g,con_b)[0]
# im_l_save=crop(saved_1,f"Bi_H00 {t} con L")

saved_1=saveim(cor_r,cor_g,cor_b)[0]
im_l_save=crop(saved_1,f"Bi_H0_180 {t} cor L")

# phase_r,phase_g,phase_b=convToGray(v_phase_r, v_phase_g, v_phase_b, t)
# saved_1,saved_2,saved_3,saved_4=saveim(phase_r,phase_g,phase_b)
# im_l_save=crop(saved_1,f"Bi_VH0 {t} Sv L")

# phase_r,phase_g,phase_b=convToGray(h_phase_r,h_phase_g,h_phase_b, t)
# saved_1,saved_2,saved_3,saved_4=saveim(phase_r,phase_g,phase_b)
# im_l_save=crop(saved_1,f"Bi_H0 {t} Sh L")
phase_r,phase_g,phase_b=convToGray_nB(h_phase_r,h_phase_g,h_phase_b )
saved_1,saved_2,saved_3,saved_4=saveim(phase_r,phase_g,phase_b)
im_l_save=crop(saved_1,f"Bi_H0 {t} Sh L")

phase_r,phase_g,phase_b=convToGray(h_phase_r_fl,h_phase_g_fl,h_phase_b_fl, t)
saved_1,saved_2,saved_3,saved_4=saveim(phase_r,phase_g,phase_b)
im_l_save=crop(saved_1,f"Bi_H180 {t} Sh L")

end_t=time.time()
print(f"Time consuming {end_t-start_t}s")