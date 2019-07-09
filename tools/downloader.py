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
# # Downloader
#
# Get all relevant images from Udacity's repo

# %%
import os
import urllib.request

# %%
def create_folder_if_not_existing(save_folder):
    """Create folder if it does not already exists. Return 0 if does not exists and 1 if exists."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("Folder created.")
        return 0
    else:
        print("Folder called {} already exists".format(save_folder))
        return 1
        
def download(urls, save_path):
    for i, url in enumerate(urls):
        save_to = os.path.join(save_path, url.split("/")[-1])
        urllib.request.urlretrieve(url, save_to)
        print("Image saved to {}".format(save_to))

# %%
def download_calibration_images(save_folder="camera_calibration_images"):
    img_urls = []
    for i in range(20):
        img_urls.append("https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/camera_cal/calibration{}.jpg".format(i+1))
    already_exists = create_folder_if_not_existing(save_folder)
    if already_exists:
        print("Already downloaded.")
    else:
        download(img_urls, save_folder)
    print("Done.")

# %%
def download_test_images(save_folder="../assets"):
    img_urls = []
    for i in range(2):
        img_urls.append("https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/test_images/test{}.jpg".format(i+1))
        img_urls.append("https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/test_images/straight_lines{}.jpg".format(i+1))
    already_exists = create_folder_if_not_existing(save_folder)  
    if already_exists:
        print("Already downloaded.")
    else:
        download(img_urls, save_folder)
    print("Done.")

# %%


