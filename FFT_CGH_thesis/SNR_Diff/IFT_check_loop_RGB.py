# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:55:01 2024

@author: Shang Gao
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2,ifftshift
import os
"""This file can process multiple holograms together and automatically, where the SNR and Diff are calculated for each one."""
# Read the path of original image.
original_path2="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/fl_one.png"

original2=plt.imread(original_path2)
original2_r=original2[:,:,0]/np.sum(original2[:,:,0])
original2_g=original2[:,:,1]/np.sum(original2[:,:,1])
original2_b=original2[:,:,2]/np.sum(original2[:,:,2])
# Save the iteration numbers in different arrays.
r1=np.array([0])
r2=np.array([1,40]) # ,5,10,20,40
r3=np.array([20])#, 20, 25, 30, 35
power=[1, 2, 3, 4]
# n1 and n2 are iterations for the GS without and with the wimdow.
for n1 in r2:
    # r4=np.arange(5,41-n1,5)
    # r4=40-r3
    for n2 in r1:
        # n2=40-n1
        # Use n1, n2 to complete the file path for reading different holograms.
        # file_path1 = f"C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/SNR_Diff_whitecircle_TF/L300/n1 or n2/One circle_1024_GS_n1_{n1},n2_{n2}_nl n_noise.png"
        # file_path1 = f"C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/SNR_Diff small RGB and fl_TF/RGB/RGB_500_GS_n1_{n1},n2_{n2}_1,nl,p.png"
        for p in power:
            file_path1 =f"C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/SNR_Diff small RGB and fl_TF/fl/L620/fl_one_GS_n1_{n1},n2_{n2}_p_{p},nl.png"
            # Read RGB CGHs .
            holo1 = plt.imread(file_path1)
            
            height=holo1.shape[0]
            width=holo1.shape[1]
            # Convert grayscales to angles.
            holo1_R_angle=holo1[:,:,0]*2*np.pi
            holo1_G_angle=holo1[:,:,1]*2*np.pi
            holo1_B_angle=holo1[:,:,2]*2*np.pi
            
            # Compute inverse FFT of phase holograms to get reconstrutions.
            Reconstruction_holo1_r = ifftshift(ifft2(np.exp(1j * holo1_R_angle)))
            Reconstruction_holo1_g = ifftshift(ifft2(np.exp(1j * holo1_G_angle)))
            Reconstruction_holo1_b = ifftshift(ifft2(np.exp(1j * holo1_B_angle)))
              
            I1_r=np.abs(Reconstruction_holo1_r)**2
            I1_r_normalized = I1_r / np.sum(I1_r) # All the sum is 1.
            I1_g=np.abs(Reconstruction_holo1_g)**2
            I1_g_normalized = I1_g / np.sum(I1_g)
            I1_b=np.abs(Reconstruction_holo1_b)**2
            I1_b_normalized = I1_b / np.sum(I1_b)
            
            l=620
            c_w,c_h=width//2,height//2
            lh,lw=height-2*l,width-2*l # The size of the window.
            
            #SNR in the window.
            SNR_r=np.sum(I1_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_r)-np.sum(I1_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
            SNR_g=np.sum(I1_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_g)-np.sum(I1_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
            SNR_b=np.sum(I1_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_b)-np.sum(I1_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
            SNR=(SNR_r+SNR_g+SNR_b)/3
            
            diff_r1=original2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_r_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
            diff_g1=original2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_g_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
            diff_b1=original2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_b_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
            # Diff in the window with respect to the original.
            D1_r=np.sqrt(np.sum((diff_r1)**2))
            D1_g=np.sqrt(np.sum((diff_g1)**2))
            D1_b=np.sqrt(np.sum((diff_b1)**2))
            D1=(D1_r+D1_g+D1_b)/3
            print(f"n1 {n1} n2 {n2} p{ p}: SNR={SNR} Diff={D1}")
            
            # max_I=1e-4#0.9025
            # max_M=0.8#0.95
            
            f=500**2#2.2e2 # Small value for seeing noise.
            I1_rgb=np.zeros_like(holo1)
            I1_rgb[:, :, 0] = I1_r_normalized*f#*height*width#(factor**2)
            I1_rgb[:, :, 1] = I1_g_normalized*f#*height*width#(factor**2)
            I1_rgb[:, :, 2] = I1_b_normalized*f#*height*width#(factor**2)
            # print(f"max {np.max(I1_rgb)}")
            # Save the reconstructed intensity.
            plt.figure()
            plt.imshow(I1_rgb)
            # plt.colorbar()
            plt.axis("off")
            plt.savefig(f"Full Re I n1_{n1} n2_{n2} p_{p}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # plt.figure()
            # plt.plot(I1_rgb[c_h,:])
            # plt.show()
            # # Save the reconstructed intensity.
            # plt.figure()
            # plt.imshow(I1_rgb*100,vmax=max_I*100)
            # # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(f"Full Re I n1_{n1} n2_{n2} noise.png", dpi=300, bbox_inches='tight')
            # plt.close()
            
            # M_rgb=np.zeros_like(holo1)
            # M_rgb[:, :, 0] = np.abs(Reconstruction_holo1_r)*f
            # M_rgb[:, :, 1] = np.abs(Reconstruction_holo1_g)*f
            # M_rgb[:, :, 2] = np.abs(Reconstruction_holo1_b)*f
            
            # # plt.figure()
            # # plt.plot(M_rgb[c_h,:])
            # # plt.show()
            # # # print(f"max {np.max(M_rgb)}")
            # plt.figure()
            # plt.imshow(M_rgb)
            # # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(f"Full Re Mag n1_{n1} n2_{n2} p_{p}.png", dpi=300, bbox_inches='tight')
            # plt.close()
            
            # plt.figure()
            # plt.imshow(M_rgb*10,vmax=max_M*10)
            # # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(f"Full Re Mag n1_{n1} n2_{n2} noise.png", dpi=300, bbox_inches='tight')
            # plt.close()
            
            # max_Diff=2.3114600480766967e-05
            # # min_Diff=-2.3114600480766967e-05
            # # Save the difference distribution between reconstruction and the original, here the absolute calculation is applied.
            # plt.figure()
            # plt.imshow(abs(diff_r1)+abs(diff_g1)+abs(diff_b1),vmax=max_Diff,cmap="YlGnBu")
            # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(f"Diff n1_{n1} n2_{n2}.png", dpi=300, bbox_inches='tight')
            # plt.close() 
           
            # Save the intensity profile crossing the center row.
            plt.figure()
            plt.plot(I1_r_normalized[c_h,:], label="R channel", color='red')
            plt.plot(I1_g_normalized[c_h,:], label="G channel", color='green')
            plt.plot(I1_b_normalized[c_h,:], label="B channel", color='blue')
            plt.ylim(0,2.5e-5)
            plt.legend()
            plt.gca().xaxis.set_visible(False)
            plt.savefig(f"I profile n1_{n1} n2_{n2} p_{p}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # plt.figure()
            # plt.plot(np.abs(Reconstruction_holo1_r)[c_h,:], label="R channel", color='red')
            # plt.plot(np.abs(Reconstruction_holo1_g)[c_h,:], label="G channel", color='green')
            # plt.plot(np.abs(Reconstruction_holo1_b)[c_h,:], label="B channel", color='blue')
            # plt.ylim(0,0.005)
            # plt.legend()
            # plt.gca().xaxis.set_visible(False)
            # plt.savefig(f"Mag profile n1_{n1} n2_{n2} p_{p}.png", dpi=300, bbox_inches='tight')
            # plt.close()
I_o=np.zeros_like(holo1)
I_o[:, :, 0] = original2_r*f#*(factor**2)
I_o[:, :, 1] = original2_g*f#*(factor**2)
I_o[:, :, 2] = original2_b*f#*(factor**2)
# plt.figure()
# plt.imshow(I_o,vmax=max_I)
# plt.colorbar()
# plt.axis("off")
# plt.savefig(f"Full Or.png", dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure()
# plt.plot(I_o[c_h,:])
# plt.show()

# M_o=np.zeros_like(holo1)
# M_o[:, :, 0] = np.sqrt(original2_r)*f
# M_o[:, :, 1] = np.sqrt(original2_g)*f
# M_o[:, :, 2] = np.sqrt(original2_b)*f

# M_o=np.zeros_like(holo1)
# M_o[:, :, 0] = np.sqrt(original2[:,:,0])
# M_o[:, :, 1] = np.sqrt(original2[:,:,1])
# M_o[:, :, 2] = np.sqrt(original2[:,:,2])

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
plt.plot(np.sqrt(original2_r)[c_h,:], label="R channel", color='red')
plt.plot(np.sqrt(original2_g)[c_h,:], label="G channel", color='green')
plt.plot(np.sqrt(original2_b)[c_h,:], label="B channel", color='blue')
plt.ylim(0,0.005)
plt.legend()
plt.gca().xaxis.set_visible(False)
plt.savefig(f"M profile Or.png", dpi=300, bbox_inches='tight')
plt.close()