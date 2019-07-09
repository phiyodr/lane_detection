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
# # Camera Calibration

# %%
import numpy as np
import cv2
from glob import glob as gglob
import matplotlib.pyplot as plt
import pickle
from os.path import join

# %%
class CameraCalibration():
    """Class to calibrate a camera using chessboard images from different angles.
    
    >>> cc = CameraCalibration("camera_calibration_folder", img_format="jpg", nx=6, ny=9)
    >>> cc.calibrate()
    >>> cc.save("calibration.p")
    """
    def __init__(self, img_folder, img_format, nx, ny):
        self.img_folder = img_folder
        self.nx = nx
        self.ny = ny
        self.img_names = self._get_all_img_names(img_format)
        
    def _get_all_img_names(self, img_format = "jpg"):
        """Get all image names from folder."""
        img_names = gglob(join(self.img_folder, '*.' + img_format))
        if len(img_names) < 1:
            print("No images in folder called {}.".format(path))
        return img_names

    def _make_object_point_array(self):
        """Array containing grid for chessboard corner points."""
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.ny, 0:self.nx].T.reshape(-1, 2)
        return objp

    def _collect_image_points(self, verbose):
        """Collect all image points (corners) from chessboard."""
        objp = self._make_object_point_array()

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane. = corners

        success = 0
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(self.img_names):
            img = cv2.imread(fname)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(img_gray, (self.ny, self.nx), None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                success += 1
                if verbose == True:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (self.ny, self.nx), corners, ret)
                    plt.imshow(img)
                    plt.title("{}: {}".format(idx, fname))
                    plt.show()

            else:
                print("Cound not process idx={} fname={}".format(idx, fname))
        if success < 20:
                print("WARN: Only {} successfull processed images. Please consider to use more images.".format(success))
        self.success = success
        return objpoints, imgpoints
    
    def _calibrate_using_objpoints_and_imgpoints(self, objpoints, imgpoints):
        """Calibrate using object and image points."""
        img = cv2.imread(self.img_names[0])
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                           img_size, None, None)
        self.mtx = mtx
        self.dist = dist
        
    def calibrate(self, verbose=False):
        """Calibrate"""
        try:
            objpoints, imgpoints = self._collect_image_points(verbose)
            self._calibrate_using_objpoints_and_imgpoints(objpoints, imgpoints)
            print("Calibration successful.")
        except:
            print("ERROR: Calibration failed.")

    def save(self, path="dist_pickle.p"):
        "Save "
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open(path, "wb" ))
        print("Calibration result saved in {}".format(path))

# %%
def undistort_image(img, mtx=None, dist=None, pkl_name=None):
    """Undistort image using mtx and dist or loading saved values from pickle (pkl_name)."""
    if pkl_name:
        dist_pickle = pickle.load(open(pkl_name, "rb" ))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    img_undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return img_undistorted

# %%

