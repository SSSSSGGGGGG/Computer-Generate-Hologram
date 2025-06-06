# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:55:45 2025

@author:Shang Gao 
"""

# import matplotlib.pyplot as plt
# import numpy as np


# h,w=50,100
# rn_p=np.random.uniform(0,1,(h,w))
# rn_m=np.random.randint(0, 2, size=(h, w))
# rn_c=1-rn_m

# # plt.figure()
# # plt.imshow(rn_p,cmap="hsv")
# # # plt.colorbar()
# # plt.show()
# # plt.imsave(f"rand_phase.png", rn_p,cmap='hsv')

# plt.figure()
# plt.imshow(rn_m,cmap="gray")
# # plt.colorbar()
# plt.show()
# # plt.imsave(f"rand_amplitude.png", rn_m,cmap="gray")

# plt.figure()
# plt.imshow(rn_c,cmap="gray")
# # plt.colorbar()
# plt.show()
# # plt.imsave(f"rand_com.png", rn_m,cmap="gray")

# import numpy as np
# import matplotlib.pyplot as plt

# h, w = 100,100

# # Random integers from 0, 1, 2 for 3 complementary regions
# label_mask = np.random.randint(0, 3, size=(h, w))

# # Generate three binary masks
# mask_0 = (label_mask == 0).astype(int)
# mask_1 = (label_mask == 1).astype(int)
# mask_2 = (label_mask == 2).astype(int)

# # Visualize them
# plt.figure()
# plt.imshow(mask_0, cmap='gray')
# # plt.title("Mask 0")
# plt.show()

# plt.figure()
# plt.imshow(mask_1, cmap='gray')
# # plt.title("Mask 1")
# plt.show()

# plt.figure()
# plt.imshow(mask_2, cmap='gray')
# # plt.title("Mask 2")
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Image size (should be even to make 2×2 blocks easy)
h, w = 100, 100

# Make sure h and w are even
assert h % 2 == 0 and w % 2 == 0, "h and w must be even for 2x2 blocks"

# Step 1: Generate random labels for blocks
block_labels = np.random.randint(0, 3, size=(h // 2, w // 2))  # One label per 2x2 block

# Step 2: Expand each label to 2x2 region
label_mask = np.kron(block_labels, np.ones((4, 4), dtype=int))  # Repeat each label in 2x2 block

# Step 3: Create binary masks
mask_0 = (label_mask == 0).astype(int)
mask_1 = (label_mask == 1).astype(int)
mask_2 = (label_mask == 2).astype(int)

# plt.figure()
# plt.imshow(block_labels, cmap="gray")
# # plt.title("Mask 0")
# plt.show()

# plt.figure()
# plt.imshow(label_mask, cmap="gray")
# # plt.title("Mask 1")
# plt.show()

# plt.figure()
# plt.imshow(mask_0, cmap="gray")
# plt.title("Mask 0")
# plt.show()

# plt.figure()
# plt.imshow(mask_1, cmap="gray")
# plt.title("Mask 1")
# plt.show()

# plt.figure()
# plt.imshow(mask_2, cmap="gray")
# plt.title("Mask 2")
# plt.show()

# Save if needed
plt.imsave("mask_0.png", mask_0, cmap="gray")
plt.imsave("mask_1.png", mask_1, cmap="gray")
plt.imsave("mask_2.png", mask_2, cmap="gray")