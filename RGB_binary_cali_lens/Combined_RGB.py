# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_1=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/1_GS_It1_20_It2_0_L_p 1.png")
im_2=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/2_GS_It1_20_It2_0_L_p 1.png")
im_3=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/3_GS_It1_20_It2_0_L_p 1.png")
im_4=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/4_GS_It1_20_It2_0_L_p 1.png")
im_5=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/5_GS_It1_20_It2_0_L_p 1.png")

im_1_array = np.array(im_1)*255
im_2_array = np.array(im_2)*255
im_3_array = np.array(im_3)*255
im_4_array = np.array(im_4)*255
im_5_array = np.array(im_5)*255
# Create a new array for the new image with the same shape as the original
im_new_array = np.zeros_like(im_1_array)
im_new_array[:,:,0] = im_1_array[:,:,0]+im_2_array[:,:,0]+im_3_array[:,:,0]+im_4_array[:,:,0]+im_5_array[:,:,0]
im_new_array[:,:,1] = im_1_array[:,:,1]+im_2_array[:,:,1]+im_3_array[:,:,1]+im_4_array[:,:,1]+im_5_array[:,:,1]
im_new_array[:,:,2] = im_1_array[:,:,2]+im_2_array[:,:,2]+im_3_array[:,:,2]+im_4_array[:,:,2]+im_5_array[:,:,2]

im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('Dear_3d.png')
im_new.show()
"""RGB"""
# im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGB_binary_cali_lens/R__184(V)_p70.png")
# im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGB_binary_cali_lens/G__136(V)_p54.png")
# im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGB_binary_cali_lens/B__112(V)_p48.png")

# im_rh_array = np.array(im_rh)*255
# im_gh_array = np.array(im_gh)*255
# im_bh_array = np.array(im_bh)*255
# # Create a new array for the new image with the same shape as the original
# im_new_array = np.zeros_like(im_rh_array)
# im_new_array[:,:,0] = im_rh_array[:,:,0]
# im_new_array[:,:,1] = im_gh_array[:,:,1]
# im_new_array[:,:,2] = im_bh_array[:,:,2]

# im_new_array = im_new_array.astype(np.uint8)

# im_new = Image.fromarray(im_new_array)
# im_new.save('RGB_tri_lens.png')
# im_new.show()
