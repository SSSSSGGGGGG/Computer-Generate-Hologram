# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np

file_n="B_" 
height, width=2716,2716

arr_r=np.zeros((1080, 1920))
pixel_size=4.5e-6#4.5e-6
f=2 # meters
# Define wavelengths (in meters, for example)
lambda_r = 0.450e-6 
center_h=1080//2
center_w=1920//2
"""RGB lens"""
for i in range(1080):
    for j in range(1920):
        r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
        arr_r[i, j] =  -r**2 / (f * lambda_r) #np.pi *
        
arr_r_mod=np.mod(arr_r,2)
"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*(255/3.55)
"""Convert array to image"""
rgb_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

G=80

im_blank=Image.new("RGB",(width,height))
pixels=im_blank.load()
stripe_width = 24 # period
spacing=stripe_width*2   # 2 times period

loop_h=height/stripe_width
reminder_h=height%stripe_width
# print(round(loop_w),loop_h)
# This is for vertical diffraction pattern distribution
if reminder_h==0:
    for x in range(width):
        for i in range(0, height, spacing):
            for j in range(stripe_width):
                if i+j<height:
                    pixels[x,i+j]=(0,0,G)
    # im_blank.save(f"{file_n}_{G}(V)_p{spacing}.png")
else:
    for x in range(width):
        for i in range(0, height, spacing):
            for j in range(stripe_width):
                if i+j<height:
                    pixels[x,i+j]=(0,0,G)
        i2=0
        while i2 < reminder_h-1:
          
            pixels[x,i2+1+(round(loop_h)-1)*stripe_width]=(0,0,G) 
            # print(i2+1+(round(loop_w)-1)*stripe_width)
            i2+=1
    # im_blank.save(f"{file_n}_{G}(V)_p{spacing}.png")
rotated_image = im_blank.rotate(45, expand=True)  # `expand=True` adjusts the output image size to fit the rotated image

h,w=rotated_image.size
center_w,center_h=w//2,h//2
cropped_rotated_image=np.array(rotated_image)[center_h-1080//2:center_h+1080//2,center_w-1920//2:center_w+1920//2]
# rgb_cropped_rotated_image = Image.fromarray(cropped_rotated_image, 'RGB')
# rgb_cropped_rotated_image.save(f"{file_n}_{G}(D)_p{spacing}.png")

rgb_image[:, :, 2] = np.clip(arr_r_modified + cropped_rotated_image[:, :, 2], 0, 255)
rgb_image_pil = Image.fromarray(rgb_image, 'RGB')
rgb_image_pil.save(f"{file_n}_{G}(D)_p{spacing}.png")

