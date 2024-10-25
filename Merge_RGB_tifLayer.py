# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/R_144(V)_p64.png")
im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/Green_112(V)_p54.png")
im_bh=plt.imread("B_80(V)_p46.png")
# im_gray=plt.imread("Gray(V)_p64.png")

# plt.figure(1)
# plt.imshow(im_rh)
# plt.figure(2)
# plt.imshow(im_gray)

# Convert the images to Pillow images
im_rh_pil = Image.fromarray((im_rh * 255).astype(np.uint8))

im_gh_pil = Image.fromarray((im_gh * 255).astype(np.uint8))
im_bh_pil = Image.fromarray((im_bh * 255).astype(np.uint8))
# im_grayv_pil = Image.fromarray((im_gray * 255).astype(np.uint8))
# plt.figure(1)
# plt.imshow(im_grayv_pil)
#Save images as layers in a single TIFF file
im_rh_pil.save('M_RGB_Bi_V_pi.tif', save_all=True, append_images=[im_gh_pil, im_bh_pil])

# Display the third image
# plt.figure(3)
# plt.imshow(im_rh_pil)
# plt.show()

im_m =Image.open('M_RGB_Bi_V_pi.tif')
# Number of layers in the TIFF file
num_layers = im_m.n_frames
layers = []

# Read each layer and append to the list
for i in range(num_layers):
    im_m.seek(i)  # Move to the i-th layer
    layer = im_m.copy()
    layers.append(layer)

# Display all layers using matplotlib
fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

for ax, layer, i in zip(axes, layers, range(num_layers)):
    ax.imshow(layer)
    ax.set_title(f'Layer {i + 1}')
    ax.axis('off')

plt.show()

# im_mr=im_m[:,:,0]
# plt.figure(4)
# plt.imshow(im_mr* 255)
# im_mg=im_m[:,:,1]
# im_mb=im_m[:,:,2]