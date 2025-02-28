# -*- coding: utf-8 -*-
"""
Created on February 13 2025
@author: Shang Gao, Ignacio Moreno

Simulates the temporal integration of multiple CGHs

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift
import time

start_t=time.time()

Iterations = [1, 5, 10, 15, 20, 25, 30, 35, 40]#,5, 10, 15, 20, 25, 30, 35, 40

# Read the path of original image.
original_path2="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/Ch6/One circle_1024.png"
original=plt.imread(original_path2)

Target=original[:,:,0]
original_nor=Target/np.sum(Target)
h,w=Target.shape

count = 0
for count in Iterations:
    Intensity = 0
    # print(f"Intensity={Intensity}")
    for n in range(count):
        
        # print(f"n={n},count={count}")   
        #----------------------------------
        # Creates the array of random numbers [0-1]
        # RandomImage = Image.new("L", (h,w), "black") # Defines a gray scale image
        Rand = np.random.uniform(0,1,(h,w)) 
        RandomPhase = np.exp(1j*2*np.pi*Rand)
    
        Field = Target * RandomPhase
        #----------------------------------
        # FOURIER TRANSFORM
        FT_Field = fftshift(fft2(fftshift(Field)))  
        FT_Magnitude = np.abs(FT_Field)
        FT_Phase = np.angle(FT_Field)
        #----------------------------------
        # MAGNITUDE = 1
        FT_Field = np.exp(1j*FT_Phase)
        #----------------------------------
        # INVERSE FOURIER TRANSFORM
        Field = ifftshift(ifft2(ifftshift(FT_Field)))  
        Magnitude = np.abs(Field)
        
        Intensity += np.square(Magnitude)
        
    # print(f"n={n},count={count}") 
    I_average = Intensity/count
    M_average = np.sqrt(I_average)
    
    l = 300
    c_w,c_h = w//2,h//2
    lh,lw = h-2*l,w-2*l # The size of the window.
    
    # SNR in the window to the outside noise.
    SNR_r = np.sum(I_average[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2])/(np.sum(I_average)-np.sum(I_average[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]))
    # Difference in the window with respect to the original intensity.
    I_average_nor=I_average/np.sum(I_average)
    diff_r = original_nor[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]-I_average_nor[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2]
    D_r = np.sqrt(np.sum((diff_r)**2))
    
    print(f"t {count}: SNR={SNR_r} Diff={D_r}")
  
    plt.figure()
    plt.imshow(abs(diff_r),vmax=1.6e-5,cmap="YlGnBu")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"I Diff for t_{count} avr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    #----------------------------------
    #Plots the magnitude and the intensity
    plt.figure()
    plt.imshow(M_average,vmax=0.004,cmap='hot')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f"Rec_M _count_{count} avr.png", dpi=300, bbox_inches='tight')
    plt.close()
       
    #----------------------------------
    # PLOT A PROFILE
    row = h//2   
    profile_M = M_average[row,:]
    
    plt.figure()
    plt.plot(profile_M)
    plt.gca().xaxis.set_visible(False)
    plt.ylim(0,0.005)
    plt.savefig(f"Rec_M profile_{count} avr.png", dpi=300, bbox_inches='tight')
    plt.close()
#----------------------------------
end_t=time.time()
print(f"Time consuming {end_t-start_t}s")