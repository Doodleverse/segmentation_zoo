# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
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


from .imports import *
from natsort import natsorted
import os, json
import numpy as np
from tkinter import filedialog
from tkinter import *
from glob import glob
from tqdm import tqdm

#-----------------------------------


root = Tk()
root.filename =  filedialog.askdirectory(title = "Select directory of RGB IMAGE files")
data_path = root.filename
print(data_path)
root.withdraw()


W=[]
W.append(data_path)

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = data_path,title = "Select directory of NIR IMAGE files")
data_path = root.filename
print(data_path)
root.withdraw()
W.append(data_path)


##========================================================
## COLLATE FILES INTO LISTS
##========================================================

files = []
for data_path in W:
    f = natsorted(glob(data_path+os.sep+'*.jpg'))
    if len(f)<1:
        f = natsorted(glob(data_path+os.sep+'images'+os.sep+'*.jpg'))
    files.append(f)

# number of bands x number of samples
files = np.vstack(files).T

##========================================================
## MAKING NDWI IMAGERY
##========================================================

## make  direc
wend = W[-1].split('/')[-1]
newdirec = W[-1].replace(wend,'ndwi')

try:
    os.mkdir(newdirec)
except:
    pass


for counter,f in tqdm(enumerate(files)):
    g = imread(f[0])[:,:,1].astype('float')
    nir = imread(f[1]).astype('float')
    g[g==0]=np.nan
    nir[nir==0]=np.nan
    g = np.ma.filled(g)
    nir = np.ma.filled(nir)

    if not np.shape(g)==np.shape(nir):
        gx,gy=np.shape(g)
        nx,ny=np.shape(nir)
        g = scale(g, np.maximum(gx,nx), np.maximum(gy,ny))
        nir = scale(nir, np.maximum(gx,nx), np.maximum(gy,ny))

    ndwi = np.divide(g - nir, g + nir )
    ndwi[np.isnan(ndwi)]=-1
    ndwi = rescale_array(ndwi,0,255)

    imsave(newdirec+os.sep+f[1].split(os.sep)[-1].replace('nir','ndwi'), ndwi.astype('uint8'), check_contrast=False, quality=100)
