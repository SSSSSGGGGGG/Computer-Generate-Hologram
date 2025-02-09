# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

im="C:/Users/Laboratorio/MakeHologram/FFT_CGH_thesis/SNR_Diff/fl_one_GS_n1_1,n2_0_p_2,nl.png f 1_win.png"
gray_image=plt.imread(im)
# def sharpness_laplacian(image):
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)  # Apply Laplacian filter
#     variance = laplacian.var()  # Compute variance
#     return variance

# # Load and process image

# sharpness_value = sharpness_laplacian(gray_image)
# print(f"Sharpness Score (Laplacian Variance): {sharpness_value}")

# def sharpness_tenengrad(image):
#     sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Gradient magnitude
#     return magnitude.mean()  # Average gradient strength

# sharpness_value = sharpness_tenengrad(gray_image)
# print(f"Sharpness Score (Tenengrad): {sharpness_value}")

# def edge_density(image):
#     edges = cv2.Canny(image.astype(np.uint8), 100, 200)
#     return edges.sum() / edges.size  # Fraction of edge pixels

# sharpness_value = edge_density(gray_image)
# print(f"Sharpness Score (Edge Density): {sharpness_value}")

def high_freq_energy(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    high_freq_energy = np.sum(magnitude_spectrum[image.shape[0]//4:, image.shape[1]//4:])  # Ignore low frequencies
    return high_freq_energy

sharpness_value = high_freq_energy(gray_image)
print(f"Sharpness Score (High-Frequency Energy): {sharpness_value}")
