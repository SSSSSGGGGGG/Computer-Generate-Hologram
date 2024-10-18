# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_rh=plt.imread("C:/Users/Laboratorio/MakeHologram/tri_blue is center/R_bl2_bi_tri_p200.png")
im_gh=plt.imread("C:/Users/Laboratorio/MakeHologram/G_bl2_tri_p300.png")
im_bh=plt.imread("C:/Users/Laboratorio/MakeHologram/tri/B_tri_112(V)_p48.png")
# im_gray=plt.imread("Gray(V)_p64.png")

im_rh_array = np.array(im_rh)*255
im_gh_array = np.array(im_gh)*255
im_bh_array = np.array(im_bh)*255
# Create a new array for the new image with the same shape as the original
im_new_array = np.zeros_like(im_rh_array)

# Modify the new array
im_new_array[:,:,0] = im_rh_array[:,:,0]
im_new_array[:,:,1] = im_gh_array[:,:,1]
im_new_array[:,:,2] = im_bh_array[:,:,2]

im_new_array = im_new_array.astype(np.uint8)

im_new = Image.fromarray(im_new_array)
im_new.save('RGB_bl2_V_tri_s_newgreen.png')


im_new.show()
# plt.figure(2)
# plt.imshow(im_gray)

# # Convert the images to Pillow images
# im_rh_pil = Image.fromarray((im_rh * 255).astype(np.uint8))

# im_gh_pil = Image.fromarray((im_gh * 255).astype(np.uint8))
# im_bh_pil = Image.fromarray((im_bh * 255).astype(np.uint8))
# im_grayv_pil = Image.fromarray((im_gray * 255).astype(np.uint8))
# # plt.figure(1)
# # plt.imshow(im_grayv_pil)
# #Save images as layers in a single TIFF file
# im_rh_pil.save('multi_layer_image.tiff', save_all=True, append_images=[im_gh_pil, im_bh_pil,im_grayv_pil])

# # Display the third image
# # plt.figure(3)
# # plt.imshow(im_rh_pil)
# # plt.show()

# im_m =Image.open('multi_layer_image.tiff')
# # Number of layers in the TIFF file
# num_layers = im_m.n_frames
# layers = []

# # Read each layer and append to the list
# for i in range(num_layers):
#     im_m.seek(i)  # Move to the i-th layer
#     layer = im_m.copy()
#     layers.append(layer)

# # Display all layers using matplotlib
# fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

# for ax, layer, i in zip(axes, layers, range(num_layers)):
#     ax.imshow(layer)
#     ax.set_title(f'Layer {i + 1}')
#     ax.axis('off')

# plt.show()

# # im_mr=im_m[:,:,0]
# # plt.figure(4)
# # plt.imshow(im_mr* 255)
# # im_mg=im_m[:,:,1]
# # im_mb=im_m[:,:,2]