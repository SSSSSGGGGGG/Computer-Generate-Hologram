# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/Lens/R_bi_l_bl.png")
im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/Lens/G_bi_l_bl.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/Lens/B_bi_l_bl.png")

im_rh_array = np.array(im_rh)*255
im_gh_array = np.array(im_gh)*255
im_bh_array = np.array(im_bh)*255
# Create a new array for the new image with the same shape as the original
im_new_array = np.zeros_like(im_rh_array)
im_new_array[:,:,0] = im_rh_array[:,:,0]
im_new_array[:,:,1] = im_gh_array[:,:,1]
im_new_array[:,:,2] = im_bh_array[:,:,2]

im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('RGB_tri_lens.png')
im_new.show()