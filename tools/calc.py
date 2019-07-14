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
# # calc

# %%
import numpy as np

# %%
def calc_dist_to_center(img_width, left_fitx, right_fitx):
    """Calculate the distance to the center of the street."""
    center = img_width / 2.
    left_bottom_x = left_fitx[-1]
    right_bottom_x = right_fitx[-1]

    lane_width = right_bottom_x - left_bottom_x
    center_lane = (lane_width / 2.0) + left_bottom_x
    cms_per_pixel = 3.7 / lane_width # US steet width is approx. 3.7 m
    dist_to_center = (center_lane - center) * cms_per_pixel
    return dist_to_center

# %%
def measure_curvature_real(ploty, left_fitx, right_fitx):
    """Calculates the curvature of polynomial functions in meters."""
    # Define conversions in x and y from pixels space to meters
    dist_in_pix = right_fitx[-1] - left_fitx[-1]
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/dist_in_pix # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    y_eval = np.max(ploty) 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad
