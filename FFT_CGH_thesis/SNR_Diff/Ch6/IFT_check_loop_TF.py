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
original_path2="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6/One circle_1024.png"

original2=plt.imread(original_path2)
original2_r=original2[:,:,0]/np.sum(original2[:,:,0])
original2_g=original2[:,:,1]/np.sum(original2[:,:,1])
original2_b=original2[:,:,2]/np.sum(original2[:,:,2])
# Save the iteration numbers in different arrays.
r1=np.array([0])
r2=np.array([ 1, 5, 10, 15, 20, 25, 30, 35,40]) # 1, 5, 10, 15, 20, 25, 30, 35,
r3=np.array([5, 10, 15, 20, 25, 30, 35])
# n1 and n2 are iterations for the GS without and with the wimdow.
for n1 in r3:
    r4=np.arange(5,41-n1,5)
    for n2 in r4:
        # Use n1, n2 to complete the file path for reading different holograms.
        # file_path1 = f"C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/SNR_Diff_whitecircle_TF/L300/n1 or n2/One circle_1024_GS_n1_{n1},n2_{n2}_nl n_noise.png"
        file_path1 = f"C:/Users/Laboratorio/OneDrive/Documents/Microstar/Simulation of difference of CGHs/SNR_Diff_whitecircle_TF/L300/n1 and n2/One circle_1024_GS_n1_{n1},n2_{n2}_nl.png"
        
        # Automatically retrieve image names
        holo1_name = os.path.basename(file_path1)
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
        
        l=300 
        c_w,c_h=width//2,height//2
        lh,lw=height-2*l,width-2*l # The size of the window.
        
        #SNR in the window.
        SNR_r=np.sum(I1_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_r)-np.sum(I1_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
        SNR_g=np.sum(I1_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_g)-np.sum(I1_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
        SNR_b=np.sum(I1_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I1_b)-np.sum(I1_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
        # SNR=(D1_r_snr+D1_g_snr+D1_b_snr)/3
        diff_r1=original2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_r_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
        diff_g1=original2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_g_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
        diff_b1=original2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I1_b_normalized[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
        # Diff in the window with respect to the original.
        D1_r=np.sqrt(np.sum((diff_r1)**2))
        D1_g=np.sqrt(np.sum((diff_g1)**2))
        D1_b=np.sqrt(np.sum((diff_b1)**2))
        # D1=(D1_r+D1_g+D1_b)/3
        print(f"n1 {n1} n2 {n2}: SNR={SNR_r} Diff={D1_r}")
        
        # Save the reconstructed intensity.
        plt.figure()
        plt.imshow(I1_r_normalized,cmap="hot")
        plt.colorbar()
        plt.savefig(f"Full Re I n1_{n1} n2_{n2}.png", dpi=300, bbox_inches='tight')
        plt.close() 
        
        factor=0.05 # Small value for seeing noise.
        I1_rgb=np.zeros_like(holo1)
        I1_rgb[:, :, 0] = I1_r_normalized/(np.max(abs(I1_r_normalized)*factor))
        I1_rgb[:, :, 1] = I1_g_normalized/(np.max(abs(I1_g_normalized)*factor))
        I1_rgb[:, :, 2] = I1_b_normalized/(np.max(abs(I1_b_normalized)*factor))
        # Save the adjusted reconstructed intensity, where the noise is emphasized.
        plt.figure()
        plt.imshow(np.clip(I1_rgb[:, :, 0], 0, 1),cmap="hot")
        plt.colorbar()
        plt.axis("off")
        plt.savefig(f"Full Re n1_{n1} n2_{n2} vf {factor}.png", dpi=300, bbox_inches='tight')
        plt.close() 
        # plt.show()
        # Save the difference distribution between reconstruction and the original, here the absolute calculation is applied.
        plt.figure()
        plt.imshow(abs(diff_r1),cmap="hot")
        plt.colorbar()
        plt.savefig(f"Diff n1_{n1} n2_{n2}.png", dpi=300, bbox_inches='tight')
        plt.close() 
        # plt.show()
        # Save the intensity profile crossing the center row.
        plt.figure()
        plt.plot(I1_r_normalized[c_h,:])
        plt.savefig(f"I profile n1_{n1} n2_{n2}.png", dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()
