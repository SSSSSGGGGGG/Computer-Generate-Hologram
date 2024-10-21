# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/GrayScript/R_bl2_tri_p200.png")

im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/GrayScript/G_bl2_tri_p320.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/GrayScript/B_tri_112(V)_p48.png")

im_rh_array = np.array(im_rh)*255
im_gh_array = np.array(im_gh)*255
im_bh_array = np.array(im_bh)*255

im_m = np.zeros_like(im_rh_array)
im_m[:,:,0]=im_rh_array[:,:,0]
im_m[:,:,1]=im_gh_array[:,:,1]
im_m[:,:,2]=im_bh_array[:,:,2]

im_int=im_m.astype(int)

im_new_array = np.zeros_like(im_rh_array)
im_new_array = im_int
im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('gray_tri_bl2_gp320.png')

im_new.show()