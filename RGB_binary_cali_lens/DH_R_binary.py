# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np

file_n="R_" 
height, width=1080,1920
arr_r=np.zeros((height, width))
pixel_size=4.5e-6#4.5e-6
f=2 # meters
# Define wavelengths (in meters, for example)
lambda_r = 0.633e-6  # Red wavelength
center_h=height//2
center_w=width//2
"""RGB lens"""
for i in range(height):
    for j in range(width):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        arr_r[i, j] =  -r**2 / (f * lambda_r) #np.pi *
        
arr_r_mod=np.mod(arr_r,2)
"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*(255/1.85)
"""Convert array to image"""
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

G=[176]
# G=[  0 , 16 , 32 , 48 , 64 , 80 , 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]#92   
def holo(G):
    G=int(G)
    print(G)
      
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
        # Convert im_blank to a NumPy array to allow array operations
    im_blank_array = np.array(im_blank)

    # Combine the red channel of `im_blank_array` with `arr_r_modified`
    rgb_image[:, :, 0] = np.clip(arr_r_modified + im_blank_array[:, :, 0], 0, 255)

    # Convert `rgb_image` back to a PIL image and save it
    rgb_image_pil = Image.fromarray(rgb_image, 'RGB')
    rgb_image_pil.save(f"{file_n}_{G}(V)_p{spacing}.png")
for i in range(len(G)):
    holo(G[i])                    
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

# im_blank.show()
# im_blank_h.show()
