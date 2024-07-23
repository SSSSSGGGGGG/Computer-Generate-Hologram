# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt

k=2
delta=4.5e-6
phi=0.64*np.pi
# sigma = 10e-4
shift=0
# the binary phase grating
def y_v(x_):
    
    
        if (0<=abs(x_)<0.5) :
            
            return np.exp(1j * phi)
        else:
           
            return 1
        
def y_v_s(x_):
        s=1.25
    
        if (0+s<=abs(x_)<0.5+s) :
            
            return np.exp(1j * phi)
        else:
           
            return 1
       
# The comb func
def comb(x_):
    if isinstance(x_, float) and x_.is_integer():# if isinstance(x_, float) and x_.is_integer(): x_==5.0
        return 1
    else:
       
        return 0
        

        
x = np.arange(0,10.1,0.1)
x2 = np.arange(0,1,0.01)
x3= np.arange(1.25,2.25,0.01)

# the input gaussian beam 
# gaussian = np.exp(-(x-5)**2 / (2 * sigma**2))
# print(x)


y=[y_v(xp) for xp in x2]
y_shift=[y_v_s(xp) for xp in x3]
y_comb=[comb(xp) for xp in x]

convolution=np.convolve(y,y_comb,mode='same')



# combined=gaussian*convolution # the intensity is 0

fft_result = np.fft.fftshift(np.fft.fft(y))
k = np.fft.fftshift(np.fft.fftfreq(len(y), 0.01))

fft_result_s = np.fft.fftshift(np.fft.fft(y_shift))
k_s = np.fft.fftshift(np.fft.fftfreq(len(y_shift), 0.01))




fig, ax = plt.subplots(1, 3,figsize=(12, 6))

ax[0].plot(x2,y, label='Approximation of rect Function')
ax[0].set_title("Binary phase grating", loc="center")
ax[1].plot(k_s, np.abs(fft_result_s)**2, label='Gaussian')#np.abs(fft_result)**2
ax[1].set_title("blazed grating", loc="center")
ax[1].plot(k,np.abs(fft_result)**2, label='FT or intensity')
ax[1].set_title("FT or intensity", loc="center")
ax[1].set_label("on")
ax[2].plot(x3,y_shift, label='FT or intensity')
ax[2].set_title("shift", loc="center")
# ax[2].plot(x,y_comb, label='comb')
# plt.figure(2)
# plt.imshow(np.abs(fft_result)**2)
# plt.colorbar(label='hotmap')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Define a function to approximate the delta function
# def delta_approx(x, epsilon=0.1):
#     if abs(x) < epsilon:
#         return 1.0 / (2 * epsilon)
#     else:
#         return 0.0

# # Generate x values
# x = np.linspace(-1, 1, 400)
# y = [delta_approx(xi, epsilon=0.05) for xi in x]

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(x, y, label='Approximation of Delta Function')
# plt.title('Approximation of Dirac Delta Function')
# plt.xlabel('x')
# plt.ylabel('Delta Function Value')
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.legend()
# plt.grid(True)
# plt.show()


