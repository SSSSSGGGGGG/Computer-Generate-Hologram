# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np


# # create LUT of calibration with respect to g=0
# # phase={0:(0,0,0),1.86*np.pi:(255,0,0)}
# phase_group={}
# for i in range(0,256,1):
#     interval=-1.86/255
#     phase={i*interval:i} # the unit in key is pi
#     phase_group.update(phase)
# # dict_group=dict(phase_group[i, for i in range (255)])


# def find_closest_key(d, search_key, tolerance):
#     closest_key = None
    
#     for key in d.keys():
#         if abs(key - search_key) < tolerance:
#             closest_key = key
#             break  # Stop early if within tolerance
#     return closest_key

# # Example usage
# search_key = -1.61
# tolerance = 0.005  # Adjust the tolerance as needed
# closest_key = find_closest_key(phase_group, search_key, tolerance)
# if closest_key is not None:
#     print(f"Closest key: {closest_key}")
#     print(f"Value: {phase_group[closest_key]}")
# else:

#     print("No key found within the tolerance level.")

# The key point is the period which is also in line with const=lamda/period
# km=m/p, m is order, and p is period.
# xm=m*d*lamda/period, where d is the distance between SLM and camera
# beacause of p, we could have a realtion between xm and km, xm=km*lamda*d
# the phase change is up to 2pi=255
    
file_n="B"   
width,height=1920,1080
im_blank=Image.new("RGB",(width,height))
pixels=im_blank.load()
stripe_width = 182 # period
spacing=stripe_width*2   # 2 times period
# this corresponds to phase value, now we need it to be 2pi.
color_value=160
# This is for vertical diffraction pattern distribution
# for x in range(width):
#     for i in range(0, height, spacing):
#         for j in range(stripe_width):
#             if i+j<height:
#                 pixels[x,i+j]=(255,0,0)
# im_blank.save(f"{file_n}(V)_p{stripe_width}.png")
                
# This is for horizontal
im_blank_h=Image.new("RGB",(width,height))
pixels_h = im_blank_h.load()

loop=width/stripe_width
reminder=width%stripe_width
# print(round(loop),reminder)
if reminder==0:
    for x1 in range(height):
        for i1 in range(0, stripe_width):
            interval=color_value/(stripe_width-1)
            for k in range(int(loop)):
                # print(i1+k*stripe_width)
                pixels_h[i1+k*stripe_width,x1]=(0,0,int((stripe_width-i1)* interval))
    im_blank_h.save(f"{file_n}(blazed)_p{stripe_width}.jpg")
else:
    interval=color_value/(stripe_width-1)
    for x1 in range(height):
        for i1 in range(0, stripe_width):
            
            for k in range(int(loop)):
                # print(i1+k*stripe_width)
                pixels_h[i1+k*stripe_width,x1]=(0,0,int((stripe_width-i1)* interval))
        i2=0
        while i2 < reminder-1:
          
            pixels_h[i2+1+(round(loop)-1)*stripe_width,x1]=(0,0,int((reminder-i2)* interval)) 
            # print(i2+1+(round(loop)-1)*stripe_width)
            i2+=1
    im_blank_h.save(f"{file_n}(blazed)_p{stripe_width}.jpg")
    
# the following is for vertical
loop_v=height/stripe_width
reminder_v=height%stripe_width
print(loop_v,reminder_v)
if reminder_v==0:
    for x1 in range(width):
        for i1 in range(0, stripe_width):
            interval=color_value/(stripe_width-1)
            for k in range(int(loop_v)):
                # print(i1+k*stripe_width)
                pixels[x1,i1+k*stripe_width]=(0,0,int((stripe_width-i1)* interval))
    im_blank.save(f"{file_n}(blazed)V_p{stripe_width}.jpg")
else:
    interval=color_value/(stripe_width-1)
    for x1 in range(width):
        for i1 in range(0, stripe_width):
            
            for k in range(int(loop_v)):
                # print(i1+k*stripe_width)
                pixels[x1,i1+k*stripe_width]=(0,0,int((stripe_width-i1)* interval))
                # print(x1,i1+k*stripe_width)
        i2=0
        while i2 < reminder_v-1:
            # print(round(loop))
            pixels[x1,i2+1+(round(loop_v)-1)*stripe_width]=(0,0,int((reminder_v-i2)* interval)) 
            
            i2+=1
    im_blank.save(f"{file_n}(blazed)V_p{stripe_width}.jpg")
im_blank.show()
im_blank_h.show()
