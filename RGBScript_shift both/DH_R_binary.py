# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np



G=92   
file_n="R_tri"   
width,height=1920,1080
im_blank=Image.new("RGB",(width,height))
pixels=im_blank.load()
stripe_width = 35 # period
spacing=stripe_width*2   # 2 times period
# This is for vertical diffraction pattern distribution

loop_w=width/stripe_width
reminder_w=width%stripe_width
# print(round(loop_w),reminder_w)
loop_h=height/stripe_width
reminder_h=height%stripe_width
# print(round(loop_w),loop_h)
# This is for vertical diffraction pattern distribution
if reminder_w==0:
    for x in range(width):
        for i in range(0, height, spacing):
            for j in range(stripe_width):
                if i+j<height:
                    pixels[x,i+j]=(G,0,0)
    im_blank.save(f"{file_n}_{G}(V)_p{spacing}.png")
else:
    for x in range(width):
        for i in range(0, height, spacing):
            for j in range(stripe_width):
                if i+j<height:
                    pixels[x,i+j]=(G,0,0)
        i2=0
        while i2 < reminder_w-1:
          
            pixels[x,i2+1+(round(loop_h)-2)*stripe_width]=(0,0,0) 
            # print(i2+1+(round(loop_w)-1)*stripe_width)
            i2+=1
    im_blank.save(f"{file_n}_{G}(V)_p{spacing}.png")
                
# # This is for horizontal
# im_blank_h=Image.new("RGB",(width,height))
# pixels_h = im_blank_h.load()
# if reminder_h==0:
#     for x1 in range(height):
#         for i1 in range(0, width, spacing):
#             for j1 in range(stripe_width):
#                 if i1+j1<width:
#                     pixels_h[i1+j1,x1]=(G,0,0)
#     im_blank_h.save(f"{file_n}_{G}(H)_p{spacing}.png")
# else:
#     for x1 in range(height):
#         for i1 in range(0, width, spacing):
#             for j1 in range(stripe_width):
#                 if i1+j1<width:
#                     pixels_h[i1+j1,x1]=(G,0,0)
#         i2=0
#         while i2 < reminder_h-1:
          
#             pixels_h[i2+1+(round(loop_w)-1)*stripe_width,x1]=(0,0,0) 
#             # print(i2+1+(round(loop_w)-1)*stripe_width)
#             i2+=1
#     im_blank_h.save(f"{file_n}_{G}(H)_p{spacing}.png")

im_blank.show()
# im_blank_h.show()
