# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np


# create LUT of calibration with respect to g=0
# phase={0:(0,0,0),1.86*np.pi:(255,0,0)}
phase_group={}
for i in range(0,256,1):
    interval=-1.86/255
    phase={i*interval:i} # the unit in key is pi
    phase_group.update(phase)
# dict_group=dict(phase_group[i, for i in range (255)])


def find_closest_key(d, search_key, tolerance):
    closest_key = None
    
    for key in d.keys():
        if abs(key - search_key) < tolerance:
            closest_key = key
            break  # Stop early if within tolerance
    return closest_key

# Example usage
search_key = -1.61
tolerance = 0.005  # Adjust the tolerance as needed
closest_key = find_closest_key(phase_group, search_key, tolerance)
if closest_key is not None:
    print(f"Closest key: {closest_key}")
    print(f"Value: {phase_group[closest_key]}")
else:
    print("No key found within the tolerance level.")
    
    
G=92    
file_n="R"   
width,height=1920,1080
im_blank=Image.new("RGB",(width,height))
pixels=im_blank.load()
stripe_width = 64 # period
spacing=stripe_width*2   # 2 times period
# This is for vertical diffraction pattern distribution
for x in range(width):
    for i in range(0, height, spacing):
        for j in range(stripe_width):
            if i+j<height:
                pixels[x,i+j]=(G,0,0)
im_blank.save(f"{file_n}_{G}(V)_p{stripe_width}.png")
                
# This is for horizontal
im_blank_h=Image.new("RGB",(width,height))
pixels_h = im_blank_h.load()
for x1 in range(height):
    for i1 in range(0, width, spacing):
        for j1 in range(stripe_width):
            if i1+j1<width:
                pixels_h[i1+j1,x1]=(G,0,0)
im_blank_h.save(f"{file_n}_{G}(H)_p{stripe_width}.png")

im_blank.show()
im_blank_h.show()
