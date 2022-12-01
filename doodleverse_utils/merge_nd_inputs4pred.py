# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-2022, Marda Science LLC
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


# utility to merge multiple coincident jpeg images into nd numpy arrays
from .imports import *
import os, json
from natsort import natsorted
from skimage.io import imread
import numpy as np
from tkinter import filedialog, messagebox
from tkinter import *
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

#-----------------------------------

#-----------------------------------
root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory for storing OUTPUT files")
output_data_path = root.filename
print(output_data_path)
root.withdraw()

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = output_data_path,title = "Select FIRST directory of IMAGE files")
data_path = root.filename
print(data_path)
root.withdraw()

W=[]
W.append(data_path)

result = 'yes'
while result == 'yes':
    result = messagebox.askquestion("More directories of images?", "More directories of images?", icon='warning')
    if result == 'yes':
        root = Tk()
        root.filename =  filedialog.askdirectory(initialdir =data_path,title = "Select directory of image files")
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

###================================================

##========================================================
## NON-AUGMENTED FILES
##========================================================

## make non-aug subset first
# cycle through pairs of files and labels
for counter,f in enumerate(files):
    im=[] # read all images into a list
    for k in f:
        im.append(imread(k))
    datadict={}
    im=np.dstack(im)# create a dtack which takes care of different sized inputs

    datadict['arr_0'] = im.astype(np.uint8)
    datadict['num_bands'] = im.shape[-1]
    datadict['files'] = [fi.split(os.sep)[-1] for fi in f]
    ROOT_STRING = f[0].split(os.sep)[-1].split('.')[0]
    segfile = output_data_path+os.sep+ROOT_STRING+'_noaug_nd_data_000000'+str(counter)+'.npz'
    np.savez_compressed(segfile, **datadict)
    del datadict, im

#-----------------------------------
def load_npz(example):
    with np.load(example.numpy()) as data:
        image = data['arr_0'].astype('uint8')
        #image = standardize(image)
        file = [''.join(f) for f in data['files']]
    return image, file[0]

@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_dataset_multiclass(example):
    """
    "read_seg_dataset_multiclass(example)"
    This function reads an example from a npz file into a single image and label
    INPUTS:
        * dataset example object (filename of npz)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """
    image, file = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.string])

    return image, file

###================================

##========================================================
## READ, VERIFY and PLOT NON-AUGMENTED FILES
##========================================================
BATCH_SIZE = 1

filenames = tf.io.gfile.glob(output_data_path+os.sep+'*_noaug*.npz')
dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

print('{} files made'.format(len(filenames)))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
dataset = dataset.map(read_seg_dataset_multiclass, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
dataset = dataset.prefetch(AUTO)

try:
    os.mkdir(output_data_path+os.sep+'noaug_sample')
except:
    pass

print('.....................................')
print('Printing examples to file ...')

counter=0
for imgs,files in dataset.take(len(filenames)):

  for count,(im, file) in enumerate(zip(imgs, files)):

     im = rescale_array(im.numpy(), 0, 1)
     if im.shape[-1]:
         im = im[:,:,:3]

     plt.imshow(im)

     file = file.numpy()

     plt.axis('off')
     plt.title(file)
     plt.savefig(output_data_path+os.sep+'noaug_sample'+os.sep+ ROOT_STRING + 'noaug_ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     plt.close('all')
     counter += 1
