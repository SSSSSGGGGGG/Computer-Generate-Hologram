# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rgbLens=plt.imread("C:/Users/Laboratorio/MakeHologram/Lens/Lens_rgb.png")
im_r=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/R_tri_92(V)_p70.png")
"""Read green channel and multiplied by 255"""
im_rgbLens_=im_rgbLens[:,:,0]*255
im_r_=im_r[:,:,0]*255
"""Binary and Lens"""
im_bi_lens = np.zeros_like(im_rgbLens)
im_bi_lens[:,:,0]=im_rgbLens_+im_r_
im_bi_l = im_bi_lens.astype(np.uint8)
im_bi_l = Image.fromarray(im_bi_l)
im_bi_l.save('R_bi_lens.png')
im_bi_l.show()

"""Read blazed gratings for green"""
im_b_bls_h=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/R(blazed)_p79.png")
im_b_bls_h_=im_b_bls_h[:,:,0]*255
im_b_bls_v=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/R(blazed)V_p400.png")
im_b_bls_v_=im_b_bls_v[:,:,0]*255

"""Binary, blazeds and Lens"""
im_bi_lens_bl = np.zeros_like(im_rgbLens)
im_bi_lens_bl[:,:,0]=im_rgbLens_+im_r_+im_b_bls_h_+im_b_bls_v_
im_bi_l_bl = im_bi_lens_bl.astype(np.uint8)
im_bi_l_bl = Image.fromarray(im_bi_l_bl)
im_bi_l_bl.save('R_bi_l_bl.png')
im_bi_l_bl.show()