# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rgbLens=plt.imread("C:/Users/Laboratorio/MakeHologram/Lens/Lens_rgb_noM.png")
im_b=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/B_tri_112(V)_p46.png")
"""Read blue channel and multiplied by 255"""
im_rgbLens_=im_rgbLens[:,:,2]*255
im_b_=im_b[:,:,2]*255
"""Binary and Lens"""
im_bi_lens = np.zeros_like(im_rgbLens)
im_bi_lens[:,:,2]=im_rgbLens_+im_b_
im_bi_l = im_bi_lens.astype(np.uint8)
im_bi_l = Image.fromarray(im_bi_l)
im_bi_l.save('B_bi_lens_noM.png')
im_bi_l.show()

"""Read blazed gratings for blue"""
im_b_bls=plt.imread("C:/Users/Laboratorio/MakeHologram/RGBScript_shift both/B(blazed)_p92_163.png")
im_b_bls_=im_b_bls[:,:,2]*255
"""Binary, blazeds and Lens"""
im_bi_lens_bl = np.zeros_like(im_rgbLens)
im_bi_lens_bl[:,:,2]=im_rgbLens_+im_b_+im_b_bls_
im_bi_l_bl = im_bi_lens_bl.astype(np.uint8)
im_bi_l_bl = Image.fromarray(im_bi_l_bl)
im_bi_l_bl.save('B_bi_l_bl_noM.png')
im_bi_l_bl.show()
