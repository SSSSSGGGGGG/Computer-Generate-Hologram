# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import matplotlib.pyplot as plt
import numpy as np

im1=plt.imread("C:/Users/Laboratorio/OneDrive/Documents/Microstar/holos for TF/results/flowers-tf/IMG_gs_1.JPG")
im2=plt.imread("C:/Users/Laboratorio/OneDrive/Documents/Microstar/holos for TF/results/flowers-tf/IMG_gs_50.JPG")

# Calculate the crop dimensions
min_height = min(im1.shape[0], im2.shape[0])
min_width = min(im1.shape[1], im2.shape[1])

# Crop both images
im1_cropped = im1[:min_height, :min_width, :]
im2_cropped = im2[:min_height, :min_width, :]


Diff1=(im1_cropped+im2_cropped)
Diff1=Diff1/np.max(Diff1)

print(f"I1_avg {np.average(im1)}, I2_avg {np.average(im2)}")
# plt.figure()
# plt.imshow(im1)
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow(im2)
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow(Diff1)
# # plt.colorbar()
# plt.axis("off")
# plt.show()