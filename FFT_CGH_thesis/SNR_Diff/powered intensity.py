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
im=plt.imread(f"{filename}.png")[8:580,8:580]

im_r=im[:,:,0]/np.sum(im[:,:,0])
im_g=im[:,:,1]/np.sum(im[:,:,1])
im_b=im[:,:,2]/np.sum(im[:,:,2])

height=im.shape[0]
width=im.shape[1]
l=250
ofs=0
c_w=width//2+ofs
c_h=height//2+ofs

powers = [1, 2, 3, 4]  # List of power values
Ave_set = []

for p in powers:  # Iterate over powers, not power
    rand = np.random.uniform(0, 1, (height, width))
    rand_2pi = 2 * np.pi * rand
    p_im_r = np.sqrt(im[:, :, 0] ** p)
    p_im_g = np.sqrt(im[:, :, 1] ** p)
    p_im_b = np.sqrt(im[:, :, 2] ** p)

    p_set = [np.average(p_im_r), np.average(p_im_g), np.average(p_im_b)]
    Ave_set.append(p_set)

# Convert to NumPy array for plotting
array2d = np.array(Ave_set)

# Ensure dimensions match before plotting
if len(powers) == array2d.shape[0]:  
    plt.figure()
    plt.plot(powers, array2d[:, 0], label="R", color="r", marker="o")
    plt.plot(powers, array2d[:, 1], label="G", color="g", marker="o")
    plt.plot(powers, array2d[:, 2], label="B", color="b", marker="o")
    plt.xticks(powers)
    plt.xlabel("Power (p)")
    plt.ylabel("Average Magnitude")
    plt.title("Effect of Power on Average Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"Dimension mismatch: powers has {len(powers)} elements, but array2d has shape {array2d.shape}")

    
    # plt.figure()
    # plt.imshow(im**power)
    # # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Full I p_{power}.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    # plt.figure()
    # plt.imshow((im**power)[:,:,1], cmap="Greens")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Full I p_{power} g.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    # plt.figure()
    # plt.imshow((im**power)[:,:,2], cmap="Blues")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Full I p_{power} b.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
   

    # FT_r=fftshift(fft2(fftshift(p_im_r)))
    # FT_g=fftshift(fft2(fftshift(p_im_g)))
    # FT_b=fftshift(fft2(fftshift(p_im_b)))

    # FT_r_m=np.abs(FT_r)
    # FT_g_m=np.abs(FT_g)
    # FT_b_m=np.abs(FT_b)
    
    # iFr=ifftshift(ifft2(np.exp(1j*np.angle(FT_r))))
    # iFg=ifftshift(ifft2(np.exp(1j*np.angle(FT_g))))
    # iFb=ifftshift(ifft2(np.exp(1j*np.angle(FT_b))))
    
    # iFT_r_m=np.abs(iFr)
    # iFT_g_m=np.abs(iFg)
    # iFT_b_m=np.abs(iFb)
    
    # f=200 # Small value for seeing noise.
    # M_rgb=np.zeros((height,width,3))
    # M_rgb[:, :, 0] = iFT_r_m*f
    # M_rgb[:, :, 1] = iFT_g_m*f
    # M_rgb[:, :, 2] = iFT_b_m*f
    
    # plt.figure()
    # plt.plot(M_rgb[c_h,:])# 
    # plt.show()
    
    # # plt.figure()
    # # plt.imshow(M_rgb[:, :, 2],cmap="Reds")# M_rgb[c_h,:]
    # # plt.show()
    # # # print(f"max {np.max(M_rgb)}")
    # plt.figure()
    # plt.imshow(M_rgb)
    # # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Full Re Mag p_{power}.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    # plt.figure()
    # plt.plot(iFT_r_m[c_h,:], label="R channel", color='red')
    # plt.plot(iFT_g_m[c_h,:], label="G channel", color='green')
    # plt.plot(iFT_b_m[c_h,:], label="B channel", color='blue')
    # plt.ylim(0,0.02)
    # plt.legend()
    # plt.gca().xaxis.set_visible(False)
    # plt.savefig(f"Mag profile p_{p}.png", dpi=300, bbox_inches='tight')
    # plt.close()



# f=100
# M_o=np.zeros((height,width,3))
# M_o[:, :, 0] = np.sqrt(im_r)*f
# M_o[:, :, 1] = np.sqrt(im_g)*f
# M_o[:, :, 2] = np.sqrt(im_b)*f

# plt.figure()
# plt.plot(M_o[c_h,:])
# plt.show()

# plt.figure()
# plt.imshow(M_o)
# plt.axis("off")
# # plt.colorbar()
# plt.savefig(f"Full Mag Or.png", dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure()
# plt.plot(original2_r[c_h,:], label="R channel", color='red')
# plt.plot(original2_g[c_h,:], label="G channel", color='green')
# plt.plot(original2_b[c_h,:], label="B channel", color='blue')
# plt.ylim(0,2.5e-5)
# plt.legend()
# plt.gca().xaxis.set_visible(False)
# plt.savefig(f"I profile Or.png", dpi=300, bbox_inches='tight')
# plt.close()

plt.figure()
plt.plot(np.sqrt(im_r)[c_h,:], label="R channel", color='red')
plt.plot(np.sqrt(im_g)[c_h,:], label="G channel", color='green')
plt.plot(np.sqrt(im_b)[c_h,:], label="B channel", color='blue')
plt.ylim(0,0.02)
plt.legend()
plt.gca().xaxis.set_visible(False)
plt.savefig(f"M profile Or.png", dpi=300, bbox_inches='tight')
plt.close()
# crop_1=P1[2][c_h-l:c_h+l,c_w-l:c_w+l]
# crop_2=P2[2][c_h-l:c_h+l,c_w-l:c_w+l]
# crop_3=P3[2][c_h-l:c_h+l,c_w-l:c_w+l]
# crop_4=P4[2][c_h-l:c_h+l,c_w-l:c_w+l]
