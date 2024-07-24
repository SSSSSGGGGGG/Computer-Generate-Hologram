# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/RB_bl2.png")
# im_gh=plt.imread("Green(H)_p54.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/Gray_tri_(V)_p92.png")

im_rh_array = np.array(im_rh)
im_bh_array = np.array(im_bh)

im_m = np.zeros_like(im_rh_array)
im_m[:,:,0]=(im_rh_array[:,:,0]+im_bh_array[:,:,0])*0.5
im_m[:,:,1]=im_bh_array[:,:,1]
im_m[:,:,2]=(im_rh_array[:,:,2]+im_bh_array[:,:,2])*0.5
im_check=im_m*255 #!!!!!!!!!!!
# im_check_h=im_rh_array[:,:,0]*255
# im_check_v=im_bh_array[:,:,0]*255
im_int=im_check.astype(int)

im_new_array = np.zeros_like(im_rh_array)
im_new_array = im_int
im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('Gray_bl2_bi_tri.png')

im_new.show()