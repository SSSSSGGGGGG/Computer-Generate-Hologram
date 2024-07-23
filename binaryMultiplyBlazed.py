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
            
            return np.exp(1j * phi)*np.exp(1j*shift*x_)
        else:
           
            return 1*np.exp(1j*shift*x_)
       
# The comb func
def comb(x_):
    if isinstance(x_, float) and x_.is_integer():# if isinstance(x_, float) and x_.is_integer(): x_==5.0
        return 1
    else:
       
        return 0
        
def blz(*x_):
    interval=2*np.pi/len(*x_)
    print(interval,len(*x_))
    y=[]
    for i in range(len(*x_)):
        y_=np.exp(1j * i*interval)
        y.append(y_)
    return y
        
x = np.arange(0,10.1,0.1)
x2 = np.arange(0,1,0.1)
x3= np.arange(0,1.01,0.01)

# the input gaussian beam 
# gaussian = np.exp(-(x-5)**2 / (2 * sigma**2))
# print(x)

y_b=blz(x3)
y=[y_v(xp) for xp in x2]

y_comb=[comb(xp) for xp in x]

convolution=np.convolve(y,y_comb,mode='same')

combined=convolution*y_b

# combined=gaussian*convolution # the intensity is 0

fft_result = np.fft.fftshift(np.fft.fft(combined))
k = np.fft.fftshift(np.fft.fftfreq(len(combined), 0.1))
print(k)



fig, ax = plt.subplots(1, 3,figsize=(12, 6))

ax[0].plot(x,convolution, label='Approximation of rect Function')
ax[0].set_title("Binary phase grating", loc="center")
ax[1].plot(x3, combined, label='Gaussian')#np.abs(fft_result)**2
ax[1].set_xlim(0,10)
ax[1].set_title("blazed grating", loc="center")
ax[2].plot(k,np.abs(fft_result)**2, label='FT or intensity')
ax[2].set_title("FT or intensity", loc="center")
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


