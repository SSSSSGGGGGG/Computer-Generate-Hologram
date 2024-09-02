from PIL import Image
import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import os

# Set the working directory and load the image
os.chdir("C:/Users/gaosh/Documents/python/Digital-hologram/OriginalImage")
filename = "1.jpg"

# Load the image using PIL to enable cropping
im_o = Image.open(filename)

# Get the dimensions of the original image
width, height = im_o.size

# Calculate the coordinates to crop the 1080x1080 region from the center
left = (width - 1080) // 2
top = (height - 1080) // 2
right = left + 1080
bottom = top + 1080

# Crop the image to the 1080x1080 region
im_o = im_o.crop((left, top, right, bottom))

# Convert the cropped image to a NumPy array
im = np.array(im_o)

# Extract the red, green, and blue channels
im_r = im[:,:,0]
im_g = im[:,:,1]
im_b = im[:,:,2]

# Perform FFT and shift
im_r_fft = fftshift(fft2(im_r))
im_g_fft = fftshift(fft2(im_g))
im_b_fft = fftshift(fft2(im_b))

# Get the dimensions of the cropped image
rows, cols = im_r.shape

# Frequency coordinates
u = np.fft.fftshift(np.fft.fftfreq(rows))
v = np.fft.fftshift(np.fft.fftfreq(cols))

# Compute the phase
phase_r = np.angle(im_r_fft) + np.pi
phase_g = np.angle(im_g_fft) + np.pi
phase_b = np.angle(im_b_fft) + np.pi

# Scale phase to 0-255 range
interval_r = 255 / (2 * np.pi)
angle_0_pi_r = phase_r * interval_r

interval_g = 200 / (2 * np.pi)
angle_0_pi_g = phase_g * interval_g

interval_b = 160 / (2 * np.pi)
angle_0_pi_b = phase_b * interval_b

# Combine the channels back
im_new = np.zeros_like(im)
im_new[:,:,0] = np.mod(angle_0_pi_r, 255)
im_new[:,:,1] = np.mod(angle_0_pi_g, 255)
im_new[:,:,2] = np.mod(angle_0_pi_b, 255)
im_new_array = im_new.astype(np.uint8)

# Save the image
output_image = Image.fromarray(im_new_array)
output_filename = f"ft_of_{filename}_center_cropped.png"
output_image.save(output_filename)

# Optional: Display the image
plt.figure(figsize=(6, 6))
plt.imshow(im_new_array, cmap="hot", origin='lower')
plt.axis('off')
plt.show()
