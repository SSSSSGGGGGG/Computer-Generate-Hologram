# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

k=2
delta=4.5e-6
phi=0.637*np.pi
# sigma = 10e-4
shift=0
# the binary phase grating
def y_v(x_):
    
    
        if (x_<0) :
            
            return np.exp(1j * phi)
        else:
           
            return 1
       
# The comb func
def comb(x_,p):
    # print(x_ % 64)
    if x_ % (p) == 0:# if and (isinstance(x_, float) and x_.is_integer())
        return 1
    else:
       
        return 0
        
def comb2(x_,p):
    # print(x_ % 64)
    if x_ % (p) == 0:# if and (isinstance(x_, float) and x_.is_integer())
        return 1
    else:
       
        return 0

def blz(x_, p):
    
    period_blazed=p  # the period can be changed
    interval=2*np.pi/period_blazed
    # print(interval,period_blazed,x_ * interval)
    
    return np.exp(1j * x_ * interval)
    

x1= np.arange(0,64,1) # for blazed
pr=len(x1)
y_b=[blz(xp,pr) for xp in x1]  # blazed

x2 = np.arange(0,1920,1)  # comb
y_comb2=[comb2(xp,pr) for xp in x2]   # comb

convolution_bl=np.convolve(y_b,y_comb2,mode='same') # the sequence of binary

       
x = np.arange(0,1920,1)  # comb
x3 = np.arange(-32,32,1)
# print(x3)
pr2=len(x3)
# print(pr2)

y=[y_v(xp) for xp in x3]    #binary

y_comb=[comb(xp,pr2) for xp in x]   # comb

convolution_bi=np.convolve(y,y_comb,mode='same') # the sequence of binary


fft_result_binary = np.fft.fftshift(np.fft.fft(convolution_bi))
k_br = np.fft.fftshift(np.fft.fftfreq(len(convolution_bi), 0.1))

fft_result_blazed = np.fft.fftshift(np.fft.fft(convolution_bl))
k_bl = np.fft.fftshift(np.fft.fftfreq(len(convolution_bl), 0.1))

index=np.argmax(fft_result_blazed)
kbl=k_bl[index]  #/4.5um is the real k in SLM
print(f"the position of the first order is the value of k: {kbl}")

combined=convolution_bi*convolution_bl
exp_com=combined/abs(combined)
real_exp=np.real(exp_com)
im_exp=np.imag(exp_com)
tan_phase=im_exp/real_exp
phase_comboned=np.arctan(tan_phase)/np.pi
fft_result_combined = np.fft.fftshift(np.fft.fft(combined))
k_co = np.fft.fftshift(np.fft.fftfreq(len(combined), 0.1))


fig, ax = plt.subplots(1, 2,figsize=(12, 6))
ax[0].plot(y, label='Binary phase grating')
ax[0].set_title("Binary phase grating", loc="center")
ax[1].plot(y_comb, label='FT of binary')#np.abs(fft_result)**2

ax[1].set_title("The comb for binary", loc="center")

fig, ax = plt.subplots(1, 2,figsize=(12, 6))
ax[0].plot(convolution_bi, label='Binary phase grating')
ax[0].set_title("Binary phase grating", loc="center")
ax[1].plot(k_br,np.abs(fft_result_binary)**2, label='FT of binary')#np.abs(fft_result)**2
ax[1].set_xlim(-1,1)
ax[1].set_title("The square of FT of binary", loc="center")


fig, ax = plt.subplots(1, 2,figsize=(12, 6))
ax[0].plot(convolution_bl, label='blazed')
ax[0].set_title("Blazed phase grating", loc="center")
ax[1].plot(k_bl,np.abs(fft_result_blazed)**2, label='Blazed phase grating')#np.abs(fft_result)**2
ax[1].set_xlim(-1,1)
ax[1].set_title("The square of FT of Blazed phase grating", loc="center")

fig, ax = plt.subplots(1, 3,figsize=(12, 6))
ax[0].plot(combined, label='muliplication')
ax[0].set_title("muliplication of phase grating", loc="center")
ax[1].plot(phase_comboned, label='Only phase')#np.abs(fft_result)**2
ax[1].set_title("The retrieved phase", loc="center")
ax[2].plot(k_co,np.abs(fft_result_combined)**2, label='Blazed phase grating')#np.abs(fft_result)**2
ax[2].set_xlim(-1,1)
ax[2].set_title("The square of FT of muliplication", loc="center")



