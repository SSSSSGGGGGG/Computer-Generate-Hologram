# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/G(blazed)_p81.png")
# im_gh=plt.imread("Green(H)_p54.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/G(blazed)V_p640.png")

im_rb=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/Green_tri_150(V)_p54.png")
# im_shift=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/G(blazed)_p108.png")

im_rh_array = np.array(im_rh)*255 # blazed_h
im_bh_array = np.array(im_bh)*255 # blazed_v
# im_shift_array= np.array(im_shift)*255 # extra shift h

im_m = np.zeros_like(im_rh_array)
im_m[:,:,1]=im_rh_array[:,:,1]+im_bh_array[:,:,1]#+im_shift_array[:,:,1]
im_check=im_m[:,:,1] #!!!!!!!!!!!
im_check_h=im_rh_array[:,:,1]
im_check_v=im_bh_array[:,:,1]
im_int=im_check.astype(int)
# im_mod=(im_m/510)*255
# im_mod=im_mod.astype(np.uint8)

im_new_array = np.zeros_like(im_rh_array)
im_new_array[:,:,1] = im_int
im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
# im_compare=Image.fromarray(im_mod)
im_new.save('RGB_G_bl2.png')

# plt.imshow(im_new_array)
im_new.show()

im_rb_array = np.array(im_rb)*255
im_m_3 = np.zeros_like(im_rh_array)
im_m_3[:,:,1]=im_rh_array[:,:,1]+im_bh_array[:,:,1]+im_rb_array[:,:,1]#+im_shift_array[:,:,1]
im_check_3=im_m_3[:,:,1]
im_int_3=im_check_3.astype(int)

im_new_arr_3 = np.zeros_like(im_rh_array)
im_new_arr_3[:,:,1] = im_int_3
im_new_arr_3=im_new_arr_3.astype(np.uint8)

im_new_3 = Image.fromarray(im_new_arr_3)
im_new_3.save('RGB_G_bl2_tri.png')

im_new_3.show()