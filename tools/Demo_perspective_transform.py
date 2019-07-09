# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo: Perspective transform

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

import perspective_transform
import camera_calibration

# %%
def cv2_imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

img = cv2_imread("../assets/straight_lines1.jpg")
img = camera_calibration.undistort_image(img, pkl_name="calibration.p")

plt.imshow(img)
_ = plt.title("Original image (undistorted)")

# %%
import masking
img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
img_white = masking.mask_white_from_LUV(img_luv, thresholds=(225, 255))

img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_yellow = masking.mask_yellow_from_LAB(img_lab)

img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
img_bright = masking.mask_bright_from_HLS(img_hls)

img_combined_binary = masking.combine_masks(img_white, img_yellow, img_bright)
plt.imshow(img_combined, cmap="gray")
_ = plt.title("Masks combined (binary)")

# %%
src = np.float32([(526, 496), (762, 496), (1016, 664), (288, 664)])
dst = np.float32([(288,  464), (996,  464), (976,  664), (288,  664)])

# %%
perspective_transform.print_polyline(img, src)

# %%
binary_warped = perspective_transform.apply_perspective_transformation(img_combined_binary, src, dst)
img_warped = perspective_transform.apply_perspective_transformation(img, src, dst)
np.save('binary_warped.npy', binary_warped)

fig, axes = plt.subplots(1,2, figsize=(12, 5))
axes[0].imshow(binary_warped, cmap="gray")
axes[0].set_title("Warped binary image")

# Transformed image
axes[1].imshow(img_warped, cmap="gray")
axes[1].set_title("Warped original image")
fig.tight_layout()

# %%

