# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# im=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/dear_GS_It1_20_It2_0_L_p 2.png")
im_1=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/CGH_o.png")
im_2=plt.imread("C:/Users/Laboratorio/MakeHologram/Con_Cor/Tri_CGH_h.png")
# im_3=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/3_GS_It1_20_It2_0_L_p 1.png")
# im_4=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/4_GS_It1_20_It2_0_L_p 1.png")
# im_5=plt.imread("C:/Users/Laboratorio/MakeHologram/For Exp in thesis/5_GS_It1_20_It2_0_L_p 1.png")

im_1_array = np.array(im_1[:,:,0])*255
im_2_array = np.array(im_2[:,:,0])*255
# im_3_array = np.array(im_3)*255
# im_4_array = np.array(im_4)*255
# im_5_array = np.array(im_5)*255
im3=im_1_array+im_2_array
im3=np.mod(im3,255)

plt.figure()
plt.imshow(im_1_array,cmap="hsv")
plt.axis("off")
plt.show()
plt.imsave("con.png", im3, cmap="hsv")

# plt.figure()
# plt.imshow(im_2_array,cmap="hsv")
# plt.show()

# # """Lens"""
# # # height, width=1080,1920
# # def lens(height, width, f):
    
# #     rdp,gdp,bdp=1.78,2.56,3.5
# #     lambda_r = 0.662e-6  
# #     lambda_g = 0.518e-6  
# #     lambda_b = 0.449e-6  
# #     arr_r=np.zeros((height, width))
# #     arr_g=np.zeros((height, width))
# #     arr_b=np.zeros((height, width))
# #     pixel_size=4.5e-6
# #     offset=0.4
# #     # f=#-2+offset # focal length (meters)
# #     fg=f
# #     fb=f
# #     center_h=height//2
# #     center_w=width//2
# #     # Calculate the phase of lenses according to Eq. (9.1) for three wavelengths.
# #     for i in range(height):
# #         for j in range(width):
# #             r = pixel_size * np.sqrt((i - center_h)**2 + (j - center_w)**2)
# #             arr_r[i, j] =  r**2 / (f * lambda_r) 
# #             arr_g[i, j] =  r**2 / (fg * lambda_g)
# #             arr_b[i, j] =  r**2 / (fb * lambda_b)
# #     # Change the range of the phase into [0,2].
# #     arr_r_mod=np.mod(arr_r,2)
# #     arr_g_mod=np.mod(arr_g,2)
# #     arr_b_mod=np.mod(arr_b,2)
# #     # Map the phase to grayscale by corresponding phase modulation.
# #     arr_r_modified=arr_r_mod*(255/rdp)
# #     arr_g_modified=arr_g_mod*(255/gdp)
# #     arr_b_modified=arr_b_mod*(255/bdp)
    
# #     return arr_r_modified,arr_g_modified,arr_b_modified

# # lens1=lens(1080,550,-2)
# # lens2=lens(1080,550,-1.5)
# # lens3=lens(1080,550,-1)

# # blank=np.zeros((1080,1920,3))
# # lens_array1 = np.stack(lens1, axis=-1)  # shape: (350, 600, 3)
# # blank[0:1080,200:200+550]=lens_array1

# # lens_array2 = np.stack(lens2, axis=-1)  # shape: (350, 600, 3)
# # blank[0:1080,600:600+550]=lens_array2

# # lens_array3 = np.stack(lens3, axis=-1)  # shape: (350, 600, 3)
# # blank[0:1080,1100:1100+550]=lens_array3



# # Create a new array for the new image with the same shape as the original
# im_new_array = np.zeros_like(im_1_array,dtype=complex)
# # # im_new_array[:,:,0] = (im_1_array[:,:,0]+lens1[0])*(im_2_array[:,:,0]+lens2[0])*(im_3_array[:,:,0]+lens3[0])*(im_4_array[:,:,0]+lens4[0])*(im_5_array[:,:,0]+lens5[0])
# # # im_new_array[:,:,1] = (im_1_array[:,:,1]+lens1[1])*(im_2_array[:,:,1]+lens2[1])*(im_3_array[:,:,1]+lens3[1])*(im_4_array[:,:,1]+lens4[1])*(im_5_array[:,:,1]+lens5[1])
# # # im_new_array[:,:,2] = (im_1_array[:,:,2]+lens1[2])*(im_2_array[:,:,2]+lens2[2])*(im_3_array[:,:,2]+lens3[2])*(im_4_array[:,:,2]+lens4[2])*(im_5_array[:,:,2]+lens5[2])
# im_new_array[:,:,0] = np.exp(1j*im_1[:,:,0])+np.exp(1j*im_2[:,:,0])+np.exp(1j*im_3[:,:,0])+np.exp(1j*im_4[:,:,0])+np.exp(1j*im_5[:,:,0])
# im_new_array[:,:,1] = np.exp(1j*im_1[:,:,1])+np.exp(1j*im_2[:,:,1])+np.exp(1j*im_3[:,:,1])+np.exp(1j*im_4[:,:,1])+np.exp(1j*im_5[:,:,1])
# im_new_array[:,:,2] = np.exp(1j*im_1[:,:,2])+np.exp(1j*im_2[:,:,2])+np.exp(1j*im_3[:,:,2])+np.exp(1j*im_4[:,:,2])+np.exp(1j*im_5[:,:,2])

# rdp,gdp,bdp=1.78,2.56,3.5
# # Final optimized phase for encoding on SLM, turn angles to grayscales.
# optimized_phase_r = np.angle(im_new_array[:,:,0])
# phase_rr_modi=(optimized_phase_r/np.pi+1)*(255/rdp)

# optimized_phase_g = np.angle(im_new_array[:,:,1])
# phase_gr_modi=(optimized_phase_g/np.pi+1)*(255/gdp)

# optimized_phase_b = np.angle(im_new_array[:,:,2])
# phase_br_modi=(optimized_phase_b/np.pi+1)*(255/bdp)

# im_new_array_new = np.zeros_like(im_1)
# im_new_array_new[:,:,0] = phase_rr_modi
# im_new_array_new[:,:,1] = phase_gr_modi
# im_new_array_new[:,:,2] = phase_br_modi

# im_new = Image.fromarray(im_new_array_new.astype(np.uint8))
# im_new.save('dear2_5 with mlens.png')
# im_new.show()
# """Reconstruction"""
# rec=fftshift(ifft2(np.exp(1j*im)))
# # plt.figure()
# # plt.imshow(abs(im_new_array[:,:,0]))
# # plt.show()

# # plt.figure()
# # plt.imshow(np.angle(im_new_array[:,:,0]))
# # plt.show()

# # plt.figure()
# # plt.imshow(abs(im_new_array[:,:,1]))
# # plt.show()

# # plt.figure()
# # plt.imshow(np.angle(im_new_array[:,:,1]))
# # plt.show()

# plt.figure()
# plt.imshow(abs(rec)**2, cmap="hot")
# plt.show()

# # plt.figure()
# # plt.imshow(np.angle(im_new_array))
# # plt.show()
# # plt.figure()
# # plt.imshow(lens3[0])
# # plt.show()

# # plt.figure()
# # plt.imshow(abs(blank))
# # plt.show()