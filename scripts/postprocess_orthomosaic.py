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

## other imports
import rasterio 
from skimage.morphology import disk
from skimage.morphology import disk, opening, dilation, area_closing, area_opening, remove_small_holes, remove_small_objects
import numpy as np

###############################################
################# INPUTS


# ## choose model implementation type
# root = Tk()
# choices = ["5000", "10000", "20000", "30000", "40000", "50000"]
# variable = StringVar(root)
# variable.set("10000")
# w = OptionMenu(root, variable, *choices)
# w.pack()
# root.mainloop()

# minblobsize = variable.get()
# print("Min. patch size : {}".format(minblobsize))

minblobsize = 10000

### user inputs
# Request the orthomosaic geotiff file
root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select label orthomosaic file",filetypes = (("geotff file","*.tif"),("jpeg file (with xml and/or wld)","*.jpg")))
label_ortho = root.filename
print(label_ortho)
root.withdraw()




### read mosaic into memory
print("Read mosaic into memory ...")
with rasterio.open(label_ortho) as dataset:
    # Read the dataset's valid data mask as a ndarray.
    label_raster = dataset.read()
    profile = dataset.profile

label_raster = np.squeeze(label_raster)

## one-hot encode to make one 2d raster per class
print("One-hot encode mosaic ...")
nR,nC = label_raster.shape
k=np.unique(label_raster.flatten())

NCLASSES = np.max(k)+1
lstack = np.zeros((nR,nC,NCLASSES), dtype=np.uint8)
lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+label_raster[...,None]-1).astype(int) 

# del label_raster
print("Removing small holes in each class label ...")
lstack0 = lstack.copy()
lstack0 = lstack0.astype(np.uint8)
lstack0[lstack0==1]=255
for i in range(NCLASSES):
    lstack0[:,:,i] = remove_small_holes(lstack0[:,:,i], area_threshold=minblobsize) #min_size=minblobsize)

print("Morphological opening on each class label ...")
opening_disk_radius = 10 ##m for a 1m naip raster
for i in range(NCLASSES):
    lstack0[:,:,i] = opening(lstack0[:,:,i], disk(opening_disk_radius)) 

## v slow

label_raster2 = np.argmax(lstack0,-1)

plt.subplot(121)
plt.imshow(label_raster[:1000,:1000]); plt.axis('off')

plt.subplot(122)
plt.imshow(label_raster2[:1000,:1000]); plt.axis('off')
plt.show()




plt.imshow(label_raster[:1000,:1000] - label_raster2[:1000,:1000]); plt.axis('off')
plt.show()