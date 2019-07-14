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

# %%
def print_polyline(img, src):
    color_blue = (0, 0, 255)
    img_polylines = np.copy(img)
    src_ints = np.int32(src).reshape((-1,1,2))
    img_polylines = cv2.polylines(img_polylines, [src_ints], True, color_blue, thickness=4)
    plt.imshow(img_polylines)


def get_perspective_transform_matrices(img, src, dst):
    """Get perspective transformation matrices."""
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_img(img, M_or_Minv):
    """Warp or unwarp image (depending if M or Minv is passed.)."""
    return cv2.warpPerspective(img, M_or_Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
