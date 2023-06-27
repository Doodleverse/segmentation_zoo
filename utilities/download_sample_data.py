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


import sys,os, time
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import requests, zipfile, io
from glob import glob

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

#### choose zenodo release
root = Tk()
choices = ['landsat_6229071', 'landsat_6230083', 'coin_6229579', 'aerial_6234122', 'aerial_6235090','ortho_6410157']
variable = StringVar(root)
variable.set('landsat_6229071')
w = OptionMenu(root, variable, *choices)
w.pack(); root.mainloop()

dataset_id = variable.get()
print("Dataset ID : {}".format(dataset_id))

zenodo_id = dataset_id.split('_')[-1]

####======================================
try:
    os.mkdir('../sample_data')
except:
    pass

try:
    os.mkdir('../sample_data/'+dataset_id)
except:
    pass

model_direc = '../sample_data/'+dataset_id

root_url = 'https://zenodo.org/record/'+zenodo_id+'/files/' 


filename='data_sample.zip'
weights_direc = model_direc + os.sep + 'data_sample'

url=(root_url+filename)
print('Retrieving data {} ...'.format(url))
outfile = model_direc + os.sep + filename

if not os.path.exists(outfile):
    print('Retrieving data {} ...'.format(url))
    download_url(url, outfile)
    print('Unzipping data to {} ...'.format(model_direc))
    with zipfile.ZipFile(outfile, 'r') as zip_ref:
        zip_ref.extractall(model_direc)

