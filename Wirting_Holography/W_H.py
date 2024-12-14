# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifft2,ifftshift

im=plt.imread("C:/Users/gaosh/Documents/python/Computer-Generate-Hologram/Wirting_Holography/3D_dear_fit_sh.png")
im_shift=fftshift(im[:,:,0])
def comput_intensity(image):
    return np.abs(image)**2

# use W gradient to improve phase
def wirtinger_phase_improve(intensity,iterations,lr,verbose=False):
    size=intensity.shape
    # generate ramdom magnitude and phase
    reconstruted_field=np.random.uniform(0.5,1.0,size)*np.exp(1j*np.random.uniform(0,2*np.pi,size))
    
    for it in range (iterations):
        # calculate intensity of the mimiced
        current_in=comput_intensity(reconstruted_field)
        
        # define a loss function: mean square error
        
        loss=np.mean((current_in-intensity)**2)
        
        # compute gradient, this is the gradient of loss respective to imaginary field
        grad=(current_in-intensity)*reconstruted_field/np.abs(reconstruted_field+1e-8)
        reconstruted_field-=lr*grad
        reconstruted_field=np.sqrt(intensity)*np.exp(1j*np.angle(reconstruted_field))
        
    return reconstruted_field

measurd_in=comput_intensity(im_shift)
measurd_in_FT=fftshift(fft2(im_shift))
phase_in=np.angle(measurd_in_FT)

final_field=wirtinger_phase_improve(measurd_in, 10, 0.01)
final_field_FT=fftshift(fft2(final_field))
phase_out=np.angle(final_field_FT)
# Plot Results
plt.figure()
# Ground truth phase

plt.title("Original Phase")
plt.imshow(phase_in, cmap="seismic")
plt.colorbar()
plt.show()


plt.figure()
# Ground truth amplitude
plt.title("Original Amplitude")
plt.imshow(np.abs(im_shift), cmap="gray")
plt.colorbar()
plt.show()


plt.figure()
# Reconstructed phase
plt.title("Reconstructed Phase")
plt.imshow(phase_out, cmap="seismic")
plt.colorbar()
plt.show()

plt.figure()
# Reconstructed amplitude
plt.title("Reconstructed Amplitude")
plt.imshow(np.abs(final_field), cmap="gray")
plt.colorbar()
plt.show()

plt.figure()
# Reconstruction error
error = np.abs(np.abs(im_shift) - final_field)
plt.title("Reconstruction Error")
plt.imshow(error, cmap="hot")
plt.colorbar()
plt.show()