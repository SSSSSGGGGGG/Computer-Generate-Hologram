# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/R(blazed)_p512.png")
# im_gh=plt.imread("Green(H)_p54.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/R(blazed)V_p512.png")

im_rb=plt.imread("C:/Users/Laboratorio/MakeHologram/R_92(V)_p132.png")
# im_gray=plt.imread("Gray(V)_p64.png")
im_rh_array = np.array(im_rh)
im_bh_array = np.array(im_bh)

im_m = np.zeros_like(im_rh_array)
im_m[:,:,0]=im_rh_array[:,:,0]+im_bh_array[:,:,0]
im_check=im_m[:,:,0]*127.5 #!!!!!!!!!!!
im_check_h=im_rh_array[:,:,0]*255
im_check_v=im_bh_array[:,:,0]*255
im_int=im_check.astype(int)

im_new_array = np.zeros_like(im_rh_array)
im_new_array[:,:,0] = im_int
im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('R_bl2_tri.png')

im_new.show()

im_rb_array = np.array(im_rb)
im_m_3 = np.zeros_like(im_rh_array)
im_m_3[:,:,0]=im_rh_array[:,:,0]+im_bh_array[:,:,0]+im_rb_array[:,:,0]
im_check_3=im_m_3[:,:,0]*(255/3)
im_int_3=im_check_3.astype(int)

im_new_arr_3 = np.zeros_like(im_rh_array)
im_new_arr_3[:,:,0] = im_int_3
im_new_arr_3=im_new_arr_3.astype(np.uint8)

im_new_3 = Image.fromarray(im_new_arr_3)
im_new_3.save('R_bl2_bi_tri.png')

im_new_3.show()