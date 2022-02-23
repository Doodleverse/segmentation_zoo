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
choices = ['landsat_6229071', 'landsat_6230083', 'coin_6229579', 'aerial_6234122', 'aerial_6235090']
variable = StringVar(root)
variable.set('landsat_6229071')
w = OptionMenu(root, variable, *choices)
w.pack(); root.mainloop()

dataset_id = variable.get()
print("Dataset ID : {}".format(dataset_id))

zenodo_id = dataset_id.split('_')[-1]

if dataset_id.startswith('landsat'):
    ## choose data (imagery) type
    root = Tk()
    choices = ['MNDWI', 'NDWI', 'RGB-NIR-SWIR', 'RGB', 'ALL']
    variable = StringVar(root)
    variable.set('MNDWI')
    w = OptionMenu(root, variable, *choices)
    w.pack(); root.mainloop()

    dataset = variable.get()
    print("Dataset type : {}".format(dataset))

elif dataset_id.startswith('coin'):
    dataset = 'RGB'

elif dataset_id.startswith('aerial'):
    dataset = 'RGB'

## choose model implementation type
root = Tk()
choices = ['BEST','ENSEMBLE']
variable = StringVar(root)
variable.set('ENSEMBLE')
w = OptionMenu(root, variable, *choices)
w.pack(); root.mainloop()

model_choice = variable.get()
print("Model implementation choice : {}".format(model_choice))

####======================================
try:
    os.mkdir('../downloaded_models')
except:
    pass

try:
    os.mkdir('../downloaded_models/'+dataset_id)
except:
    pass

model_direc = '../downloaded_models/'+dataset_id

root_url = 'https://zenodo.org/record/'+zenodo_id+'/files/' 

if dataset_id.startswith('landsat'):

    if dataset=='RGB':
        filename='rgb.zip'
        weights_direc = model_direc + os.sep + 'rgb'
    elif dataset=='RGB-NIR-SWIR':
        filename='rgb_nir_swir.zip'
        weights_direc = model_direc + os.sep + 'rgb_nir_swir'
    elif dataset=='MNDWI':
        filename='mndwi.zip'
        weights_direc = model_direc + os.sep + 'mndwi'
    elif dataset=='NDWI':
        filename='ndwi.zip'
        weights_direc = model_direc + os.sep + 'ndwi'
    elif dataset=='ALL':
        filenames=['rgb.zip','rgb_nir_swir.zip','mndwi.zip','ndwi.zip']
        model_direcs = [model_direc for f in filenames]

elif dataset_id.startswith('coin'):
    filename='rgb.zip'
    weights_direc = model_direc + os.sep + 'rgb'

elif dataset_id.startswith('aerial'):
    filename='rgb.zip'
    weights_direc = model_direc + os.sep + 'rgb'

###=================================================================

if dataset=='ALL':
    for filename, model_direc in zip(filenames, model_direcs):
        url=(root_url+filename)
        outfile = model_direc + os.sep + filename

        if not os.path.exists(outfile):
            print('Retrieving model {} ...'.format(url))
            download_url(url, outfile)
            print('Unzipping model to {} ...'.format(model_direc))
            with zipfile.ZipFile(outfile, 'r') as zip_ref:
                zip_ref.extractall(model_direc)

else:
    url=(root_url+filename)
    print('Retrieving model {} ...'.format(url))
    outfile = model_direc + os.sep + filename

    if not os.path.exists(outfile):
        print('Retrieving model {} ...'.format(url))
        download_url(url, outfile)
        print('Unzipping model to {} ...'.format(model_direc))
        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall(model_direc)

###==============================================
if dataset=='ALL':
    sys.exit(2)
else:
    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of images (or npzs) to segment")
    sample_direc = root.filename
    print(sample_direc)
    root.withdraw()

if model_choice=='ENSEMBLE':
    Ww = glob(weights_direc+os.sep+'*.h5')
    print("{} sets of model weights were found ".format(len(W)))
else:
    #read best model file
    #select weights
    with open(weights_direc+os.sep+'BEST_MODEL.txt') as f:
        w = f.readlines()
    Ww = [weights_direc + os.sep + w[0]]


sys.path.insert(1, 'src')

USE_GPU = True

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from prediction_imports import *
#====================================================


M= []; C=[]; T = []
for counter,weights in enumerate(Ww):
    configfile = weights.replace('.h5','.json').replace('weights', 'config')

    if 'fullmodel' in configfile:
        configfile = configfile.replace('_fullmodel','')


    with open(configfile) as f:
        config = json.load(f)

    for k in config.keys():
        exec(k+'=config["'+k+'"]')

    from imports import *

    try:
        model = tf.keras.models.load_model(weights)

    except:

        if MODEL =='resunet':
            model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                            FILTERS,
                            nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                            kernel_size=(KERNEL,KERNEL),
                            strides=STRIDE,
                            dropout=DROPOUT,#0.1,
                            dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                            dropout_type=DROPOUT_TYPE,#"standard",
                            use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                            )
        elif MODEL=='unet':
            model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                            FILTERS,
                            nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                            kernel_size=(KERNEL,KERNEL),
                            strides=STRIDE,
                            dropout=DROPOUT,#0.1,
                            dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                            dropout_type=DROPOUT_TYPE,#"standard",
                            use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                            )

        elif MODEL =='simple_resunet':
            # num_filters = 8 # initial filters
            # model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_filters, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

            model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        kernel = (2, 2),
                        num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        filters=FILTERS,#8,
                        num_layers=4,
                        strides=(1,1))
        #346,564
        elif MODEL=='simple_unet':
            model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        kernel = (2, 2),
                        num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        filters=FILTERS,#8,
                        num_layers=4,
                        strides=(1,1))
        #242,812

        elif MODEL=='satunet':
            #model = sat_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_classes=NCLASSES)

            model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        kernel = (2, 2),
                        num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        filters=FILTERS,#8,
                        num_layers=4,
                        strides=(1,1))
        model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [mean_iou, dice_coef])

        model.load_weights(weights)

    M.append(model)
    C.append(configfile)
    T.append(MODEL)


metadatadict = {}
metadatadict['model_weights'] = Ww
metadatadict['config_files'] = C
metadatadict['model_types'] = T


### predict
print('.....................................')
print('Using model for prediction on images ...')

sample_filenames = sorted(glob(sample_direc+os.sep+'*.*'))
if sample_filenames[0].split('.')[-1]=='npz':
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.npz'))
else:
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
    if len(sample_filenames)==0:
        # sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))
        sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

#look for TTA config
if not 'TESTTIMEAUG' in locals():
    TESTTIMEAUG = False

for f in tqdm(sample_filenames):
    do_seg(f, M, metadatadict, sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG)


# w = Parallel(n_jobs=2, verbose=0, max_nbytes=None)(delayed(do_seg)(f) for f in tqdm(sample_filenames))
