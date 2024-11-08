# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGB_binary_cali_lens/CrossTalk/R__184(V)_p48.png")
im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGB_binary_cali_lens/CrossTalk/G__136(H)_p48.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/RGB_binary_cali_lens/CrossTalk/B__80(D)_p48.png")
im_rh_array = np.array(im_rh)*255
im_gh_array = np.array(im_gh)*255
im_bh_array = np.array(im_bh)*255
"""RG"""
im_new_array = np.zeros_like(im_rh_array)
im_new_array[:,:,0] = im_rh_array[:,:,0]
im_new_array[:,:,1] = im_gh_array[:,:,1]

im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('RV_GH.tif')
im_new.show()

"""RB"""
im_new_rb = np.zeros_like(im_rh_array)
im_new_rb[:,:,0] = im_rh_array[:,:,0]
im_new_rb[:,:,2] = im_bh_array[:,:,2]

im_new_rb = im_new_rb.astype(np.uint8)

im_new_rb= Image.fromarray(im_new_rb)
im_new_rb.save('RV_BD.tif')
im_new_rb.show()

"""GB"""
im_new_gb = np.zeros_like(im_rh_array)
im_new_gb[:,:,1] = im_gh_array[:,:,1]
im_new_gb[:,:,2] = im_bh_array[:,:,2]

im_new_gb = im_new_gb.astype(np.uint8)

im_new_gb = Image.fromarray(im_new_gb)
im_new_gb.save('GH_BD.tif')
im_new_gb.show()