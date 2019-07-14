# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Masking

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ### masks

def mask_white_from_LUV(img_luv, thresholds=(225, 255)):
    """Extract the white parts of an image using L channel of an LUV image. 
    Function returns a binary mask (2D numpy array)."""
    l_channel = img_luv[:,:,0]
    channel_l_binary = np.zeros_like(l_channel)
    channel_l_binary[(l_channel >= thresholds[0]) & (l_channel <= thresholds[1])] = 1
    return channel_l_binary


def mask_yellow_from_LAB(img_lab, thresholds=(155, 200)):
    """Extract the yellow parts of an image using B channel of an LAB image. 
    Function returns a binary mask (2D numpy array)."""
    b_channel = img_lab[:,:,2]
    channel_b_binary = np.zeros_like(b_channel)
    channel_b_binary[(b_channel >= thresholds[0]) & (b_channel <= thresholds[1])] = 1
    return channel_b_binary


def mask_bright_from_HLS(img_hls, thresholds=(200, 255)):
    s_channel = img_hls[:,:,2]
    channel_s_binary = np.zeros_like(s_channel)
    channel_s_binary[(s_channel >= thresholds[0]) & (s_channel <= thresholds[1])] = 1
    return channel_s_binary


def mask_sobel_from_HLS(img_hls, deriv="x", thresholds=(20, 100)):
    """Extract x or y edges with Sobel filter using L channel of a HLS image. 
    Function returns a binary mask (2D numpy array)."""
    l_channel = img_hls[:,:,1]
    if deriv == "x":
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    elif deriv == "y":
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    else:
        print("Use 'x' or 'y' for deriv argument.")
    sobel_abs = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    sobel_scaled = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    sobel_binary = np.zeros_like(sobel_scaled)
    # Threshold gradient 
    sobel_binary[(sobel_scaled >= thresholds[0]) & (sobel_scaled <= thresholds[1])] = 1
    return sobel_binary


# ### combine functions

def combine_masks(*args): 
    """Combine several masks (2d numpy arrays) to show all active pixels."""
    combined_binary = np.zeros_like(args[0])
    print
    for arg in args:
        combined_binary[(arg == 1)] = 1
    return combined_binary


def combine_masks_in_color(*args):
    """Combine 2 or 3 masks (2d numpy arrays) to show all active pixels in RGB."""
    if len(args) == 2:
        img_color = np.dstack((args[0], args[1], np.zeros_like(args[0]))) * 255
    elif len(args) == 3:
        img_color = np.dstack((args[0], args[1], args[2])) * 255
    else:
        print("You can only pass 2 or 3 images. You passed {} images".format(len(args)))
    return img_color


def get_yellow_white_and_bright_pixel_mask(img):
    """Get a binary mask of pixels which might be lane pixels, 
    i.e. yellow, white and bright pixels."""
    # white
    img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    img_white = mask_white_from_LUV(img_luv, thresholds=(225, 255))
    # yellow
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_yellow = mask_yellow_from_LAB(img_lab)
    # bright
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_bright = mask_bright_from_HLS(img_hls)
    # combine
    img_combined_binary = combine_masks(img_white, img_yellow, img_bright)
    return img_combined_binary

def plot_thresholds(img, fct, min_list, max_list, figsize=(15, 50)):
    """Plot all min_list and max_list combinations applied to img using fct."""
    fig, axes = plt.subplots(len(min_list), len(max_list), figsize=figsize)
    for i, iv in enumerate(min_list):
        for j, jv in enumerate(max_list):
            if iv < jv:
                img_processed = fct(img, thresholds=(iv, jv))
            else:
                img_processed = np.ones(img.shape)
            axes[i, j].imshow(img_processed, cmap="gray")
            axes[i, j].set_title("(min={},max={})".format(iv, jv))
            axes[i, j].axis('off')
    #fig.tight_layout()
