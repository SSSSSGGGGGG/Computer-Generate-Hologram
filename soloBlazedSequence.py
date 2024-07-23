# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


       
def comb(x_,p):
    # print(x_ % 64)
    if x_ % (p) == 0:# if and (isinstance(x_, float) and x_.is_integer())
        return 1
    else:
       
        return 0

def blz(x_, p):
    
    period_blazed=p  # the period can be changed
    interval=3.55*np.pi/period_blazed
    # print(interval,period_blazed,x_ * interval)
    
    return np.exp(1j * x_ * interval),x_ * interval
    
# print(np.exp(1j *3.55*np.pi ))
x1= np.arange(0,46,1) # for blazed
pr=len(x1)
y_b,y_b_p=zip(*[blz(xp,pr)[:2] for xp in x1])  # blazed

x2 = np.arange(0,1920,1)  # comb
y_comb=[comb(xp,pr) for xp in x2]   # comb

convolution=np.convolve(y_b,y_comb,mode='same') # the sequence of binary



# combined=gaussian*convolution # the intensity is 0

fft_result_binary = np.fft.fftshift(np.fft.fft(convolution))
k_br = np.fft.fftshift(np.fft.fftfreq(len(convolution), 0.1))

fft_result_blazed = np.fft.fftshift(np.fft.fft(y_b))
k_bl = np.fft.fftshift(np.fft.fftfreq(len(y_b), 0.1))

index=np.argmax(fft_result_blazed)
k=k_bl[index]  #/4.5um is the real k in SLM

print(f"the position of the first order is the value of k: {k}")

fig, ax = plt.subplots(1, 3,figsize=(12, 6))
ax[0].plot(x1,y_b_p, label='phase grating')
ax[0].set_title("blazed phase grating", loc="center")
ax[1].plot(x2,y_comb, label='comb')#np.abs(fft_result)**2
# ax[1].set_xlim(-1,1)
ax[1].set_title("Comb", loc="center")
ax[2].plot(convolution, label='Convolutuion')
ax[2].set_title("Convolution", loc="center")
# ax[2].set_xlim(-1,1)

fig, ax = plt.subplots(1, 2,figsize=(12, 6))
ax[0].plot(k_bl,np.abs(fft_result_blazed)**2, label='blazed')
ax[0].set_title("FT of Blazed phase grating", loc="center")
ax[0].set_xlim(-1,1)
ax[1].plot(k_br,np.abs(fft_result_binary)**2, label='Combined phase grating')#np.abs(fft_result)**2
ax[1].set_xlim(-1,1)
ax[1].set_title("FT of convolution", loc="center")
# ax[2].plot(k_bl,np.abs(fft_result_blazed)**2, label='FT of blzaed')#np.abs(fft_result)**2
# ax[2].set_title("The square of FT of blazed grating", loc="center")
# ax[2].set_xlim(-1,1)
# ax[2].plot(k,np.abs(fft_result)**2, label='FT or intensity')
# ax[2].set_title("FT or intensity", loc="center")


