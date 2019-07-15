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
# # Lane Detection
#

# %%
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from collections import deque
from collections import OrderedDict
import pickle

# %%
from tools import camera_calibration
from tools import calc
from tools import plotting
from tools import lane_detect
from tools import masking
from tools import perspective_transform as pt


# %% [markdown]
# # Class

# %%
class LaneDetection:
    """Lane Detection."""
    def __init__(self, mtx, dist, src, dst):
        # Undistortion
        self.mtx = mtx
        self.dist = dist
        # Perspective transform
        self.src = src
        self.dst = dst
        
        # Curve fitted line
        self.left_fit = None
        self.right_fit = None
        # Curve diameter
        self.left_curve_diameter = deque(maxlen=3)
        self.right_curve_diameter = deque(maxlen=3)
        
        self.frame_nb = 0
        self.imgs = OrderedDict()

    def detect(self, img, save_interim_img=False, debug_mode=False):
        """Lane detection function."""
        if debug_mode:
            save_interim_img = True
        # Distortion correction
        img_undist = camera_calibration.undistort_image(img, self.mtx, self.dist)
        img_bright_binary = masking.get_yellow_white_and_bright_pixel_mask(img_undist)
        
        # 4. Perspective transform
        M, Minv = pt.get_perspective_transform_matrices(img_bright_binary, self.src, self.dst)
        img_binary_warped = pt.warp_img(img_bright_binary, M_or_Minv=M)
        
        # 5. Detect lane line
        # - Init run (only once)
        if self.left_fit is None:
            self.left_fit, self.right_fit, img_rectangle_warped, img_histogram = lane_detect.detect_initial_lane_line(img_binary_warped, save_interim_img)
            if save_interim_img:
                self.imgs['img_histogram'] = img_histogram
                self.imgs['img_rectangle_warped'] = img_rectangle_warped
        # - Further runs
        fits, lanes_xy, lanes_inds, lines_pts, img_lane_warped = lane_detect.detect_further_lane_line(
            img_binary_warped, self.left_fit, self.right_fit, save_interim_img)
        if save_interim_img:
                self.imgs['img_lane_warped'] = img_lane_warped
        left_fitx, right_fitx, ploty = lanes_xy
        left_fit, right_fit = fits
        left_lane_inds, right_lane_inds = lanes_inds
        left_line_pts, right_line_pts = lines_pts
        #left_points_count, right_points_count = pts_counts
        
        # Determine lane curvature
        left_curve, right_curve = calc.measure_curvature_real(ploty, left_fitx, right_fitx)
        self.left_curve_diameter.append(left_curve)
        self.right_curve_diameter.append(right_curve)
        
        # Add colored lanes and plane
        _, img_colored_lanes_warp = plotting.add_colored_lanes(img_binary_warped, left_lane_inds, right_lane_inds)
        img_colored_plane_warp = plotting.add_colored_plane(img_binary_warped, left_fitx, right_fitx, ploty)
        img_colored_warp = plotting.combine_images(img_colored_lanes_warp, img_colored_plane_warp)
        # Warp back
        img_colored_unwarp = pt.warp_img(img_colored_warp, Minv)
        img_unwarp = plotting.combine_images(img_undist, img_colored_unwarp, val1=1., val2=1.)
        if save_interim_img:
            self.imgs['img_colored_lanes_warp'] = img_colored_lanes_warp
            self.imgs['img_colored_plane_warp'] = img_colored_plane_warp
            self.imgs['img_colored_warp'] = img_colored_warp
            self.imgs['img_colored_unwarp'] = img_colored_unwarp
            self.imgs['img_unwarp'] = img_unwarp
        # Add infos
        dist_to_center = calc.calc_dist_to_center(img_unwarp.shape[1], left_fitx, right_fitx)
        img_result = plotting.add_text_values(img_unwarp, np.mean(self.left_curve_diameter), 
            np.mean(self.right_curve_diameter), dist_to_center)

        self.frame_nb += 1
        if debug_mode:
            font = cv2.FONT_HERSHEY_SIMPLEX
            color_white = (255, 255, 255)
            cv2.putText(img_lane_warped, '{:.3}, {:.3}, {:.3}'.format(*left_fit), (50, 200), font, 1.2, color_white, thickness=2)
            cv2.putText(img_lane_warped, '{:.3}, {:.3}, {:.3}'.format(*right_fit), (50, 250), font, 1.2, color_white, thickness=2)
            cv2.putText(img_lane_warped, "frame:{}".format(self.frame_nb), (50, 50), font, 1.2, color_white, thickness=2)
            #cv2.putText(img_lane_warped, "left_points:{}".format(left_points_count), (400, 50), font, 1.2, fontColor, thickness=2)
            #cv2.putText(img_lane_warped, "right_points:{}".format(right_points_count), (400, 100), font, 1.2, fontColor, thickness=2)
            return img_lane_warped
        else:
            return img_result
        
    def reset(self):
        self.left_fit = None
        self.right_fit = None
        self.left_curve_diameter = deque(maxlen=3)
        self.right_curve_diameter = deque(maxlen=3)
        self.frame_nb = 0

# %%
## Read fct, calibration values and perspective transformation parameters
#def cv2_imread(path):
#    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#calib = pickle.load(open("tools/calibration.p", "rb" ))
#src = np.float32([(526, 496), (762, 496), (1016, 664), (288, 664)])
#dst = np.float32([(288,  464), (996,  464), (976,  664), (288,  664)])
#
## Plot original and processed image
#fig, axes = plt.subplots(1,2, figsize=(12, 5))
#img = cv2_imread("assets/test2.jpg")
#axes[0].imshow(img)
#axes[0].set_title("Original image")
#
#ld = LaneDetection(calib['mtx'], calib['dist'], src, dst)
#img_res = ld.detect(img, save_interim_img=True, debug_mode=False)
#axes[1].imshow(img_res)
#_ = axes[1].set_title("Processed image")

# %%
#for name in ld.imgs:
#    shape = ld.imgs[name].shape
#    if len(shape) == 1:
#        plt.plot(ld.imgs[name])        
#    else:
#        plt.imshow(ld.imgs[name])
#    plt.title(name)
#    plt.show()

# %%

