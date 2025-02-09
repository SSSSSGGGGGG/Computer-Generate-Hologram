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

os.chdir("C:/Users/gaosh/Documents/python/Computer-Generate-Hologram/FFT_CGH_thesis/SNR_Diff")
filename="flowers_tf"  #flowers_960 RGB_1024
im=plt.imread(f"{filename}.png")

height=im.shape[0]
width=im.shape[1]


l=390 # from edge to center 250 for 3circles
c_w,c_h=width//2,height//2
lh,lw=height-2*l,width-2*l
power1=1
#R
im_shift_r=fftshift(im[:,:,0]**power1)
#G
im_shift_g=fftshift(im[:,:,1]**power1)
#B
im_shift_b=fftshift(im[:,:,2]**power1)

# Random phase generation
rand = np.random.uniform(0, 1, (height - 2 * l, width - 2 * l))
rand_2pi = 2 * np.pi * rand  # Full phase range [0, 2π]
exp_rand = np.exp(1j * rand_2pi)  # Complex exponential

# Initialize complex-valued arrays
im_n_r = np.zeros_like(im[:, :, 0], dtype=complex)
im_n_g = np.zeros_like(im[:, :, 1], dtype=complex)
im_n_b = np.zeros_like(im[:, :, 2], dtype=complex)

# Combine random phase with intensity for each channel
im_r_rand = exp_rand * np.sqrt(im[:, :, 0][c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])
im_n_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2] = im_r_rand

im_g_rand = exp_rand * np.sqrt(im[:, :, 1][c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])
im_n_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2] = im_g_rand

im_b_rand = exp_rand * np.sqrt(im[:, :, 2][c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])
im_n_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2] = im_b_rand

# Fourier Transform for each channel
current_field_r = fftshift(fft2(fftshift(im_n_r)))  # Red channel
current_field_g = fftshift(fft2(fftshift(im_n_g)))  # Green channel
current_field_b = fftshift(fft2(fftshift(im_n_b)))  # Blue channel

# #R
# im_shift_r=fftshift(im[:,:,0])
# #G
# im_shift_g=fftshift(im[:,:,1])
# #B
# im_shift_b=fftshift(im[:,:,2])


# # random
# rand=np.random.uniform(0, 1, (height, width))
# rand_2pi=np.pi*rand
# rand_ma=np.max(rand_2pi)
# rand_mi=np.min(rand_2pi)
# exp_rand=np.exp(1j*rand_2pi)
# #R
# im_r_rand=exp_rand*np.sqrt(im_shift_r)

# im_g_rand=exp_rand*np.sqrt(im_shift_g)

# im_b_rand=exp_rand*np.sqrt(im_shift_b)

# current_field_r = fftshift(fft2(im_r_rand ))
# current_field_g =fftshift(fft2(im_g_rand))
# current_field_b =fftshift(fft2(im_b_rand))
iterations1=10
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

r,g,b=abs(current_field_r),abs(current_field_g),abs(current_field_b)


def radial_profile(image, center):
    """ Compute the Radial Frequency Profile for a given Fourier magnitude image. """
    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)  # Compute radial distances

    r = r.astype(np.int32)  # Convert to integer for binning
    max_r = r.max()
    radial_sum = np.bincount(r.ravel(), weights=image.ravel())  # Sum intensities in each bin
    radial_count = np.bincount(r.ravel())  # Count elements in each bin
    radial_profile = radial_sum / np.maximum(radial_count, 1)  # Avoid division by zero

    return radial_profile[:max_r]

# Compute Radial Frequency Profile for R, G, and B channels
center = (r.shape[0] // 2, r.shape[1] // 2)
r_profile = radial_profile(r, center)
g_profile = radial_profile(g, center)
b_profile = radial_profile(b, center)

# Plot the Radial Frequency Profiles
plt.figure(figsize=(6, 4))
plt.plot(r_profile, 'r', label='Red Channel')
plt.plot(g_profile, 'g', label='Green Channel')
plt.plot(b_profile, 'b', label='Blue Channel')
plt.xlabel("Radial Frequency")
plt.ylabel("Mean Intensity")
plt.legend()
plt.title("Radial Frequency Profile")
plt.show()

# plt.figure()
# plt.imshow(r,cmap="Reds")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(g,cmap="Greens")
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(b,cmap="Blues")
# plt.colorbar()
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

# def crop(im_modify,name):
#     # y_offset=center_h-1080//2
#     im_cropped=im_modify#[y_offset:y_offset+1080,:]
#     im_cropped = im_cropped.astype(np.uint8)
#     im_modi = Image.fromarray(im_cropped)
#     im_modi.save(f"{filename}_GS_n1_{iterations1},n2_{iterations2}_{name}.png")
# # R=crop(im_modify_r, "r")
# # G=crop(im_modify_g, "g")
# # B=crop(im_modify_b, "b")
# # C=crop(im_modify_c, "c")
# C_noL=crop(im_modify_noL, f"{factor},nl,p")
# # Save each channel separately
# # red_channel.save(f"{filename}_GS_{iterations}_lens_NoCo_HA_r.png")
# # green_channel.save(f"{filename}_GS_{iterations}_lens_NoCo_HA_g.png")
# # blue_channel.save(f"{filename}_GS_{iterations}_lens_NoCo_HA_b.png")
end_t=time.time()
print(f"Time consuming {end_t-start_t}s, iteration {iterations1+iterations2}")