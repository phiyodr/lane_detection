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
# # plotting

# %%
import numpy as np
import cv2

# %%
def add_colored_plane(img_gray, left_fitx, right_fitx, ploty):
    """Add colerd plane to grayscale image."""
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_gray).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp


# %%
def add_colored_lanes(img_gray, left_lane_inds, right_lane_inds):
    """Add red lanes to grayscale image."""
    # Grab activated pixels
    nonzero = img_gray.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img_gray, img_gray, img_gray))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds],  nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    window_img[nonzeroy[left_lane_inds],  nonzerox[left_lane_inds]] = [255, 0, 0] 
    window_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0] 
    return out_img, window_img


# %%
def combine_images(img1, img2, val1=1., val2=.3):
    """Combine two images."""
    return cv2.addWeighted(img1, val1, img2, val2, 0)

# %%
def add_text_values(img, left_curverad, right_curverad, dist_to_center, average_curve_diameter=True):
    """Add curvate diameter and distance to center of steet to image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_white = (255, 255, 255)
    if average_curve_diameter:
        cv2.putText(img, 'Curvature: {:.0f} m'.format((left_curverad+ right_curverad)/2), (50, 50), font, 1.2, font_white, thickness=4)
    else:
        cv2.putText(img, 'Curvature: {:.0f} m/{:.0f} m'.format(left_curverad, right_curverad), (50, 50), font, 1.2, font_white, thickness=4)
    if dist_to_center >= 0:
        cv2.putText(img, 'Vehicle is {:.2f} m left of center'.format(dist_to_center), (50, 120), font, 1.2, font_white, thickness=4)
    if dist_to_center < 0:
        cv2.putText(img, 'Vehicle is {:.2f} m right of center'.format(-dist_to_center), (50, 120), font, 1.2, font_white, thickness=4)
    return img
