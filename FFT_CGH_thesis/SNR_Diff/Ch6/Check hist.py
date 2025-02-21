# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:55:01 2024

@author: gaosh
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift,ifft2
import os


def hist(r,g,b,name):
    # Plot histograms for each channel.
    plt.figure()   
    plt.hist(r.flatten(), bins=30, color='red', alpha=0.5, label="Red")
    plt.hist(g.flatten(), bins=30, color='green', alpha=0.5, label="Green")
    plt.hist(b.flatten(), bins=30, color='blue', alpha=0.5, label="Blue")
    
    # Add mean intensity markers.
    plt.axvline(np.average(r), color='red', linestyle='dashed', linewidth=2, label=f"Mean R={np.average(r):.10f}")
    plt.axvline(np.average(g), color='green', linestyle='dashed', linewidth=2, label=f"Mean G={np.average(g):.10f}")
    plt.axvline(np.average(b), color='blue', linestyle='dashed', linewidth=2, label=f"Mean B={np.average(b):.10f}")
    
    # Add labels, legend, and title.
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f"{name} RGB Histogram")
    plt.legend()   
    # Save the histogram plot
    plt.savefig(f"{name}_histogram.png")      
    # Show the plot
    plt.show()
# Histgram of the original image in the window.
# O_hist=hist(original2_r[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2],original2_g[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2],original2_b[c_h-lh//2:c_h+lh//2, c_w-lw//2:c_w+lw//2],f"RGB Original")
# Calculate SNR and Diff for different RGB CGHs, and save the reconstrutions by defined factor, which is designed for visulizing noise.
