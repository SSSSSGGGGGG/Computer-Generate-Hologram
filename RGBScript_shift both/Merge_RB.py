# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/R_bl2.png")
# im_gh=plt.imread("Green(H)_p54.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/G_bl2.png")
# im_gray=plt.imread("Gray(V)_p64.png")
im_rh_array = np.array(im_rh)*255
im_bh_array = np.array(im_bh)*255

im_m = np.zeros_like(im_rh_array)
im_m[:,:,0]=im_rh_array[:,:,0]
im_m[:,:,1]=im_bh_array[:,:,1]
im_check=im_m
# im_check_h=im_rh_array[:,:,0]
# im_check_v=im_bh_array[:,:,0]
im_int=im_check.astype(int)

im_new_array = np.zeros_like(im_rh_array)
im_new_array = im_int
im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('RG_bl2.png')

im_new.show()