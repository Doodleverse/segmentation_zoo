# Written by Dr Daniel Buscombe, Marda Science LLC
# for  the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE zSOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .model_imports import *
from .model_metrics import *

from glob import glob
from skimage.io import imread, imsave
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

#######===================================================
def make_dir(dirname):
    # check that the directory does not already exist
    if not os.path.isdir(dirname):
        # if not, try to create the directory
        try:
            os.mkdir(dirname)
        # if there is an exception, print to screen and try to continue
        except Exception as e:
            print(e)
    # if the dir already exists, let the user know
    else:
        print('{} directory already exists'.format(dirname))

def move_files(files, outdirec):
    for a_file in files:
        shutil.move(a_file, outdirec+os.sep+a_file.split(os.sep)[-1])


#-----------------------------------
# custom 2d resizing functions for 2d discrete labels
def scale(im, nR, nC):
  '''
  for reszing 2d integer arrays
  '''
  nR0 = len(im)     # source number of rows
  nC0 = len(im[0])  # source number of columns
  tmp = [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]
  return np.array(tmp).reshape((nR,nC))

#-----------------------------------
def scale_rgb(img, nR, nC, nD):
  '''
  for reszing 3d integer arrays
  '''
  imgout = np.zeros((nR, nC, nD))
  for k in range(3):
      im = img[:,:,k]
      nR0 = len(im)     # source number of rows
      nC0 = len(im[0])  # source number of columns
      tmp = [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]
      imgout[:,:,k] = np.array(tmp).reshape((nR,nC))
  return imgout

##========================================================
def fromhex(n):
    """hexadecimal to integer"""
    return int(n, base=16)


##========================================================
def label_to_colors(
    img,
    mask,
    alpha,  # =128,
    colormap,  # =class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,  # =0,
    do_alpha,  # =True
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """

    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask == 1] = (0, 0, 0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg


##========================================================
def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


##====================================
def standardize(img):
    # standardization using adjusted standard deviation

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img) == 2:
        img = np.dstack((img, img, img))

    return img


# ##========================================================
def inpaint_nans(im):
    ipn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans) > 0:
        im[nans] = 0
        vNeighbors = convolve2d(
            (nans == False), ipn_kernel, mode="same", boundary="symm"
        )
        im2 = convolve2d(im, ipn_kernel, mode="same", boundary="symm")
        im2[vNeighbors > 0] = im2[vNeighbors > 0] / vNeighbors[vNeighbors > 0]
        im2[vNeighbors == 0] = np.nan
        im2[(nans == False)] = im[(nans == False)]
        im = im2
        nans = np.isnan(im)
    return im


# -----------------------------------
def plot_seg_history_iou(history, train_hist_fig):
    """
    "plot_seg_history_iou(history, train_hist_fig)"
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history["val_loss"])

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(
        np.arange(1, n + 1), history.history["mean_iou"], "b", label="train accuracy"
    )
    plt.plot(
        np.arange(1, n + 1),
        history.history["val_mean_iou"],
        "k",
        label="validation accuracy",
    )
    plt.xlabel("Epoch number", fontsize=10)
    plt.ylabel("Mean IoU Coefficient", fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1, n + 1), history.history["loss"], "b", label="train loss")
    plt.plot(
        np.arange(1, n + 1), history.history["val_loss"], "k", label="validation loss"
    )
    plt.xlabel("Epoch number", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches="tight")


#-----------------------------------
def do_resize_label(lfile, TARGET_SIZE):
    ### labels ------------------------------------
    lab = imread(lfile)
    result = scale(lab,TARGET_SIZE[0],TARGET_SIZE[1])

    wend = lfile.split(os.sep)[-2]
    fdir = os.path.dirname(lfile)
    fdirout = fdir.replace(wend,'resized_'+wend)

    # save result
    imsave(fdirout+os.sep+lfile.split(os.sep)[-1].replace('.jpg','.png'), result.astype('uint8'), check_contrast=False, compression=0)


#-----------------------------------
def do_resize_image(f, TARGET_SIZE):
    img = imread(f)

    try:
        _, _, channels = img.shape
    except:
        channels=0

    if channels>0:
        result = scale_rgb(img,TARGET_SIZE[0],TARGET_SIZE[1],3)
    else:
        result = scale(img,TARGET_SIZE[0],TARGET_SIZE[1])

    wend = f.split(os.sep)[-2]
    fdir = os.path.dirname(f)
    fdirout = fdir.replace(wend,'resized_'+wend)

    # save result
    imsave(fdirout+os.sep+f.split(os.sep)[-1].replace('.jpg','.png'), result.astype('uint8'), check_contrast=False, compression=0)

