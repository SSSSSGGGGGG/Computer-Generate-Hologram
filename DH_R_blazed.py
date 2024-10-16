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
    
file_n="R"   
width,height=1920,1080
im_blank=Image.new("RGB",(width,height))
pixels=im_blank.load()
stripe_width = 256 # period
stripe_width_v = 512 # period



# this corresponds to phase value, now we need it to be 2pi.
color_value=255               
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
                pixels_h[i1+k*stripe_width,x1]=(int((i1)* interval),0,0)
    im_blank_h.save(f"{file_n}(blazed)_p{stripe_width}.png")
else:
    interval=color_value/(stripe_width-1)
    for x1 in range(height):
        for i1 in range(0, stripe_width):
            
            for k in range(int(loop)):
                # print(i1+k*stripe_width)
                pixels_h[i1+k*stripe_width,x1]=(int((i1)* interval),0,0)
        i2=0
        while i2 < reminder-1:
          
            pixels_h[i2+1+(round(loop)-1)*stripe_width,x1]=(int((i2)* interval),0,0) 
            # print(i2+1+(round(loop)-1)*stripe_width)
            i2+=1
    im_blank_h.save(f"{file_n}(blazed)_p{stripe_width}.png")

# the following is for vertical
loop_v=height/stripe_width_v
reminder_v=height%stripe_width_v
print(loop_v,reminder_v)
if reminder_v==0:
    for x1 in range(width):
        for i1 in range(0, stripe_width_v):
            interval=color_value/(stripe_width_v-1)
            for k in range(int(loop_v)):
                # print(i1+k*stripe_width)
                pixels[x1,i1+k*stripe_width_v]=(int((i1)* interval),0,0)
    im_blank.save(f"{file_n}(blazed)V_p{stripe_width_v}.png")
else:
    interval=color_value/(stripe_width_v-1)
    for x1 in range(width):
        for i1 in range(0, stripe_width_v):
            
            for k in range(int(loop_v)):
                # print(i1+k*stripe_width)
                pixels[x1,i1+k*stripe_width_v]=(int((i1)* interval),0,0)
                # print(x1,i1+k*stripe_width)
        i2=0
        while i2 < reminder_v-1:
            # print(round(loop))
            pixels[x1,i2+1+(round(loop_v)-1)*stripe_width_v]=(int((i2)* interval),0,0) 
            
            i2+=1
    im_blank.save(f"{file_n}(blazed)V_p{stripe_width_v}.png")
im_blank.show()

# im_blank.show()
im_blank_h.show()
