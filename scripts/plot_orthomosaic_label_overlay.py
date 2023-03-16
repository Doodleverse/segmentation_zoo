# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2023, Marda Science LLC
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# standard imports
from tkinter import filedialog
from tkinter import *
import sys, os, shutil, json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib


###############################################
################# INPUTS
### user inputs
# Request the orthomosaic geotiff file
root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select orthomosaic image file",filetypes = (("geotff file","*.tif"),("jpeg file (with xml and/or wld)","*.jpg"),("all files","*.*")))
image_ortho = root.filename
print(image_ortho)
root.withdraw()

# Request the orthomosaic label geotiff file
root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select label mosaic file",filetypes = (("geotff file","*.tif"),("jpeg file (with xml and/or wld)","*.jpg"),("all files","*.*")))
label_ortho = root.filename
print(label_ortho)
root.withdraw()



### read mosaic into memory
print("Read label mosaic into memory ...")
with rasterio.open(label_ortho) as dataset:
    # Read the dataset's valid data mask as a ndarray.
    label_raster = dataset.read()
    profile_label = dataset.profile

label_raster = np.squeeze(label_raster)

### read mosaic into memory
print("Read image mosaic into memory ...")
with rasterio.open(image_ortho) as dataset:
    # Read the dataset's valid data mask as a ndarray.
    image_raster = dataset.read()#([1,2,3])
    profile_image = dataset.profile

print(image_raster.dtype, np.max(image_raster))
nc, nx, ny = image_raster.shape
image_raster = image_raster.reshape((nx,ny,nc))

# image_raster = (255 * image_raster / np.max(image_raster)).astype(np.uint8)
plt.imshow(image_raster)
plt.show()

image_raster = np.squeeze(image_raster)


class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                        '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                        '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']

NUM_LABEL_CLASSES = len(np.unique(label_raster.flatten()))

class_label_colormap = class_label_colormap[:NUM_LABEL_CLASSES]
cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES+1])

#Make an overlay
plt.imshow(image_raster[:,:,:3])
plt.imshow(label_raster, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_LABEL_CLASSES)
plt.axis('off')
if 'jpg' in i:
    plt.savefig(i.replace('.jpg','_overlay.png'), dpi=200, bbox_inches='tight')
elif 'jpeg' in i:
    plt.savefig(i.replace('.jpeg','_overlay.png'), dpi=200, bbox_inches='tight')
else:
    plt.close()
