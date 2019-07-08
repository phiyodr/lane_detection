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
# # Demo

# %% [markdown]
# 0. Download images 
# 1. Masking
# 2. Camera Calibration

# %% [markdown]
# # 0. Download images 
#
# from Udacity's repo

# %%
from downloader import *

# %%
download_calibration_images("camera_calibration_images")
download_test_images("../assets")

# %% [markdown]
# # 1. Masking
#
# Example for each function.

# %%
import cv2
import masking
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ### Main functions

# %%
def cv2_imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

img = cv2_imread("../assets/test2.jpg")
plt.imshow(img)
_ = plt.title("Original image")

# %%
img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
img_white = masking.mask_white_from_LUV(img_luv, thresholds=(225, 255))
plt.imshow(img_white, cmap="gray")
_ = plt.title("White mask")

# %%
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_yellow = masking.mask_yellow_from_LAB(img_lab)
plt.imshow(img_yellow, cmap="gray")
_ = plt.title("Yellow mask")

# %%
img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
img_bright = masking.mask_bright_from_HLS(img_hls)
plt.imshow(img_bright, cmap="gray")
_ = plt.title("Bright mask")

# %%
fig, axes = plt.subplots(1,2, figsize=(12, 5))

img_combined = masking.combine_masks(img_white, img_yellow, img_bright)
axes[0].imshow(img_combined, cmap="gray")
axes[0].set_title("Masks combined (binary)")

img_combined = masking.combine_masks_in_color(img_white, img_yellow, img_bright)
axes[1].imshow(img_combined, cmap="gray")
axes[1].set_title("Masks combined (each mask one color)")
fig.tight_layout()

# %%
img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
fig, axes = plt.subplots(2,2, figsize=(12, 5))

img_sobelx = masking.mask_sobel_from_HLS(img_hls)
axes[0,0].imshow(img_sobelx, cmap="gray")
axes[0,0].set_title("Sobel: x")

img_sobely = masking.mask_sobel_from_HLS(img_hls, "y")
axes[0,1].imshow(img_sobely, cmap="gray")
axes[0,1].set_title("Sobel: y")

img_sobelxy = masking.combine_masks(img_sobely, img_sobelx)
axes[1,0].imshow(img_sobelxy, cmap="gray")
axes[1,0].set_title("Sobel: x and y (combined binary)")

img_sobelxy = masking.combine_masks_in_color(img_sobely, img_sobelx)
axes[1,1].imshow(img_sobelxy, cmap="gray")
_ = axes[1,1].set_title("Sobel: x and y (combined in color)")
fig.tight_layout()

# %% [markdown]
# ### Grid search plotting function

# %%
min_thresholds = [20, 60]   
max_thresholds = [80, 140, 200] 
masking.plot_thresholds(img_hls, masking.mask_sobel_from_HLS, 
                            min_thresholds, max_thresholds, figsize=(18,8))

# %% [markdown]
# # 2. Camera Calibration
#
#



# %%
from camera_calibration import CameraCalibration

# %%
cc = CameraCalibration("camera_calibration_images", img_format="jpg", nx=6, ny=9)
cc.calibrate()
cc.save("calibration.p")

# %%


