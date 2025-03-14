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


os.chdir("C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff")
filename="fl_one"  #flowers_960 RGB_1024

im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]



l=620 # from edge to center 250 for 3circles
c_w,c_h=width//2,height//2
lh,lw=height-2*l,width-2*l
power1=1

#R
im_shift_r=fftshift(im[:,:,0]**power1)
#G
im_shift_g=fftshift(im[:,:,1]**power1)
#B
im_shift_b=fftshift(im[:,:,2]**power1)

# random
rand=np.random.uniform(0, 1, (height, width))
rand_2pi=np.pi*rand
rand_ma=np.max(rand_2pi)
rand_mi=np.min(rand_2pi)
exp_rand=np.exp(1j*rand_2pi)
#R
im_r_rand=exp_rand*np.sqrt(im_shift_r)

im_g_rand=exp_rand*np.sqrt(im_shift_g)

im_b_rand=exp_rand*np.sqrt(im_shift_b)

current_field_r = fftshift(fft2(im_r_rand ))
current_field_g =fftshift(fft2(im_g_rand))
current_field_b =fftshift(fft2(im_b_rand))

iterations1=35
iterations2=0
factor=1
for j in range(iterations1):
    
    # Inverse Fourier Transform to initial plane
    current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
    current_field_g_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_g))))
    current_field_b_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_b))))
    
    
    current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))#*exp_rand
    current_field_g_n =np.sqrt(im_shift_g)*np.exp(1j * np.angle(current_field_g_i))#*exp_rand
    current_field_b_n =np.sqrt(im_shift_b)*np.exp(1j * np.angle(current_field_b_i))#*exp_rand
    
    # Forward Fourier Transform to the target plane
    current_field_r = fftshift(fft2(current_field_r_n ))
    current_field_g = fftshift(fft2(current_field_g_n ))
    current_field_b = fftshift(fft2(current_field_b_n ))
for i in range(iterations2):
    
    # Inverse Fourier Transform to initial plane
    current_field_r_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_r))))
    current_field_g_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_g))))
    current_field_b_i = ifft2(ifftshift(np.exp(1j * np.angle(current_field_b))))
    
    
    current_field_r_i_t =np.sqrt(abs(current_field_r_i)**2/np.max(abs(current_field_r_i)**2)) *np.exp(1j * np.angle(current_field_r_i))
    current_field_g_i_t =np.sqrt(abs(current_field_g_i)**2/np.max(abs(current_field_g_i)**2)) *np.exp(1j * np.angle(current_field_g_i))
    current_field_b_i_t =np.sqrt(abs(current_field_b_i)**2/np.max(abs(current_field_b_i)**2)) *np.exp(1j * np.angle(current_field_b_i))
    
    current_field_r_n =np.sqrt(im_shift_r)*np.exp(1j * np.angle(current_field_r_i))#*exp_rand
    current_field_g_n =np.sqrt(im_shift_g)*np.exp(1j * np.angle(current_field_g_i))#*exp_rand
    current_field_b_n =np.sqrt(im_shift_b)*np.exp(1j * np.angle(current_field_b_i))#*exp_rand
    
    current_field_r_n[:,c_w-l:c_w+l]=current_field_r_i_t[:,c_w-l:c_w+l]
    current_field_g_n[:,c_w-l:c_w+l]=current_field_g_i_t[:,c_w-l:c_w+l]
    current_field_b_n[:,c_w-l:c_w+l]=current_field_b_i_t[:,c_w-l:c_w+l]
    
    current_field_r_n[c_h-l:c_h+l,0:c_w-l]=current_field_r_i_t[c_h-l:c_h+l,0:c_w-l]
    current_field_g_n[c_h-l:c_h+l,0:c_w-l]=current_field_g_i_t[c_h-l:c_h+l,0:c_w-l]
    current_field_b_n[c_h-l:c_h+l,0:c_w-l]=current_field_b_i_t[c_h-l:c_h+l,0:c_w-l]
    
    current_field_r_n[c_h-l:c_h+l,c_w+l:width]=current_field_r_i_t[c_h-l:c_h+l,c_w+l:width]
    current_field_g_n[c_h-l:c_h+l,c_w+l:width]=current_field_g_i_t[c_h-l:c_h+l,c_w+l:width]
    current_field_b_n[c_h-l:c_h+l,c_w+l:width]=current_field_b_i_t[c_h-l:c_h+l,c_w+l:width]
    
    
    # Forward Fourier Transform to the target plane
    current_field_r = fftshift(fft2(current_field_r_n ))
    current_field_g = fftshift(fft2(current_field_g_n ))
    current_field_b = fftshift(fft2(current_field_b_n ))


# Final optimized phase for display or application on SLM
optimized_phase_r = np.angle(current_field_r)
phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/2)
# phase_rr_modi_mod=np.mod(phase_rr_modi,255)

optimized_phase_g = np.angle(current_field_g)
phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/2)
# phase_gr_modi_mod=np.mod(phase_gr_modi,255)

optimized_phase_b = np.angle(current_field_b)
phase_br_modi=(optimized_phase_b/np.pi+1)*(255/2)


r, g, b = abs(current_field_r), abs(current_field_g), abs(current_field_b)

i_sum = 0
r_sum = []
for radi in range(height // 2):
    if (c_h - radi >= 0 and c_h + radi < height) and (c_w - radi >= 0 and c_w + radi < width):  
        avg_intensity = np.mean(r[c_h + radi, c_w + radi])  # Compute mean directly
        r_sum.append(avg_intensity)  # Store the value for plotting

# plt.figure()
# plt.plot(r_sum/np.sum(r_sum))
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity')
# plt.axhline(np.average(r_sum/np.sum(r_sum)), color='red', linestyle='dashed', linewidth=2, label=f"Mean R={np.average(r_sum/np.sum(r_sum)):.10f}")
# plt.title("R ")
# plt.legend()
# plt.show()


# plt.figure()
# plt.plot(abs(r[:,959]-np.average(r[:,959])))
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity')
# plt.axhline(np.average(r[:,959]), color='red', linestyle='dashed', linewidth=2, label=f"Mean R={np.average(r[:,959]):.10f}")
# plt.title(f"R x-axis variation")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(abs(r[959,:]-np.average(r[959,:])))
# plt.axhline(np.average(r[959,:]), color='red', linestyle='dashed', linewidth=2, label=f"Mean R={np.average(r[:,959]):.10f}")
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity variation')
# plt.legend()
# plt.title(f"R y-axis")
# plt.show()

# plt.figure()
# plt.plot(abs(g[:,959]-np.average(g[:,959])))
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity variation')
# plt.axhline(np.average(g[:,959]), color='green', linestyle='dashed', linewidth=2, label=f"Mean G={np.average(g[:,959]):.10f}")
# plt.title(f"G x-axis")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(abs(g[959,:]-np.average(g[959,:])))
# plt.axhline(np.average(g[959,:]), color='green', linestyle='dashed', linewidth=2, label=f"Mean G={np.average(g[959,:]):.10f}")
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity variation')
# plt.legend()
# plt.title(f"G y-axis")
# plt.show()

# plt.figure()
# plt.plot(abs(b[:,959]-np.average(b[:,959])))
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity variation')
# plt.axhline(np.average(b[:,959]), color='blue', linestyle='dashed', linewidth=2, label=f"Mean B={np.average(b[:,959]):.10f}")
# plt.title(f"B x-axis")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(abs(b[959,:]-np.average(b[959,:])))
# plt.axhline(np.average(b[959,:]), color='blue', linestyle='dashed', linewidth=2, label=f"Mean B={np.average(b[959,:]):.10f}")
# plt.xlabel('Frequency')
# plt.ylabel('Pixel Intensity variation')
# plt.legend()
# plt.title(f"B y-axis")
# plt.show()

# plt.figure()
# # Plot histograms for each channel
# plt.hist(r.flatten(), bins=30, color='red', alpha=0.5, label='Red')
# plt.hist(g.flatten(), bins=30, color='green', alpha=0.5, label='Green')
# plt.hist(b.flatten(), bins=30, color='blue', alpha=0.5, label='Blue')

# # Add mean intensity markers
# plt.axvline(np.average(r), color='red', linestyle='dashed', linewidth=2, label=f'Mean R')
# plt.axvline(np.average(g), color='green', linestyle='dashed', linewidth=2, label=f'Mean G')
# plt.axvline(np.average(b), color='blue', linestyle='dashed', linewidth=2, label=f'Mean B')

# # Add labels, legend, and title
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')

# plt.legend()

# # Save the histogram plot
# # plt.savefig(f"{name}_histogram.png")  # Correct function to save the plot

# # Show the plot
# plt.show()
# Assuming phase_rr_modi and arr_r_modified are already defined and have matching shapes
im_modify_r = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_g = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
im_modify_b = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))

# Fill each channel with respective modifications
# im_modify_r[:,:,0] = phase_rr_modi + arr_r_modified
# im_modify_g[:,:,1]= phase_gr_modi + arr_g_modified
# im_modify_b[:,:,2] = phase_br_modi + arr_b_modified

im_modify_noL = np.zeros_like(im,shape=(im.shape[0], im.shape[1], 3))
im_modify_noL[:,:,0] = phase_rr_modi
im_modify_noL[:,:,1] = phase_gr_modi
im_modify_noL[:,:,2] = phase_br_modi

# im_modify_c = np.zeros_like(im, shape=(im.shape[0], im.shape[1],3))
# im_modify_c[:,:,0] = phase_rr_modi+arr_r_modified
# im_modify_c[:,:,1] = phase_gr_modi+arr_g_modified
# im_modify_c[:,:,2] = phase_br_modi+arr_b_modified

def crop(im_modify,name):
    # y_offset=center_h-1080//2
    im_cropped=im_modify#[y_offset:y_offset+1080,:]
    im_cropped = im_cropped.astype(np.uint8)
    im_modi = Image.fromarray(im_cropped)
    im_modi.save(f"{filename}_GS_n1_{iterations1},n2_{iterations2}_{name}.png")
# R=crop(im_modify_r, "r")
# G=crop(im_modify_g, "g")
# B=crop(im_modify_b, "b")
# C=crop(im_modify_c, "c")
C_noL=crop(im_modify_noL, f"p_{power1},nl")
# Save each channel separately
# red_channel.save(f"{filename}_GS_{iterations}_lens_NoCo_HA_r.png")
# green_channel.save(f"{filename}_GS_{iterations}_lens_NoCo_HA_g.png")
# blue_channel.save(f"{filename}_GS_{iterations}_lens_NoCo_HA_b.png")
end_t=time.time()
print(f"Time consuming {end_t-start_t}s, iteration {iterations1+iterations2}")