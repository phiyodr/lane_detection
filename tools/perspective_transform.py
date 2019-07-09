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
# # Perspective transform


# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

def print_polyline(img, src):
    blue = (0, 0, 255)
    img_polylines = np.copy(img)
    src_ints = np.int32(src).reshape((-1,1,2))
    img_polylines = cv2.polylines(img_polylines, [src_ints], True, blue, thickness=4)
    plt.imshow(img_polylines)
    
def apply_perspective_transformation(img, src, dst):
    """Apply perspective transformation using src and dst.""""
    M = cv2.getPerspectiveTransform(src, dst)
    img_warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return img_warped
