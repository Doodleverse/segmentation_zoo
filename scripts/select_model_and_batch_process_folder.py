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
import sys,os, json
import asyncio
import platform
from glob import glob

import tqdm
from tqdm.auto import tqdm as auto_tqdm
import tqdm.asyncio
import zipfile
from tkinter import filedialog
from tkinter import *
import requests
import aiohttp
import tensorflow as tf
# specify where imports come from
from doodleverse_utils.prediction_imports import do_seg

def download_zipped_model(dataset_id: str,weights_direc:str,url:str):
    # Create the directory to hold the downloaded models from Zenodo
    weights_direc = os.path.join(weights_direc,dataset_id)
    if not os.path.exists(weights_direc):
        os.mkdir(weights_direc)
    # 'rgb.zip' is name of directory containing model online
    filename='rgb'
    # outfile: full path to directory containing model files
    # example: 'c:/model_name/rgb'
    outfile = weights_direc + os.sep + filename
    if os.path.exists(outfile):   
        print(f'\n Found model weights directory: {os.path.abspath(outfile)}')
    # if model directory does not exist download zipped model from Zenodo
    if not os.path.exists(outfile):
        print(f'\n Downloading to model weights directory: {os.path.abspath(outfile)}')
        zip_file=filename+'.zip'
        zip_folder = weights_direc + os.sep + zip_file
        print(f'Retrieving model {url} ...')
        download_zip(url, zip_folder)
        print(f'Unzipping model to {weights_direc} ...')
        with zipfile.ZipFile(zip_folder, 'r') as zip_ref:
            zip_ref.extractall(weights_direc)
        print(f'Removing {zip_folder}')
        os.remove(zip_folder)
    
        
    # set weights dir to sub directory (rgb) containing model files  
    weights_direc = os.path.join(weights_direc,'rgb')
                
    # Ensure all files are unzipped
    with os.scandir(weights_direc) as it:
        for entry in it:
            if entry.name.endswith('.zip'):
                with zipfile.ZipFile(entry, 'r') as zip_ref:
                    zip_ref.extractall(weights_direc)
                os.remove(entry)
    return weights_direc
            
def download_zip(url: str, save_path: str, chunk_size: int = 128):
        """Downloads the zipped model from the given url to the save_path location.
        Args:
            url (str): url to zip directory  to download
            save_path (str): directory to save model
            chunk_size (int, optional):  Defaults to 128.
        """
        # make an HTTP request within a context manager
        with requests.get(url, stream=True) as r:
            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))
            with open(save_path, 'wb') as fd:
                with auto_tqdm(total=total_length, unit='B', unit_scale=True,unit_divisor=1024,desc="Downloading Model",initial=0, ascii=True) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            fd.write(chunk)
                            pbar.update(len(chunk))


async def fetch(session, url: str, save_path: str):
    model_name = url.split("/")[-1]
    chunk_size: int = 128
    async with session.get(url, raise_for_status=True) as r:
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with auto_tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_name}",
                    initial=0,
                    ascii=False,
                ) as pbar:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(save_path, "wb") as fd:
                async for chunk in r.content.iter_chunked(chunk_size):
                    fd.write(chunk)

async def fetch_all(session, url_dict):
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(fetch(session, url, save_path))
        tasks.append(task)
    # await tqdm.gather(tasks)
    await tqdm.asyncio.tqdm.gather(tasks)
    # await tqdm.gather(*tasks)

async def async_download_urls(url_dict: dict) -> None:
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, url_dict)

def run_async_download(url_dict: dict):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # wait for async downloads to complete
    asyncio.run(async_download_urls(url_dict))

def get_url_dict_to_download(models_json_dict: dict) -> dict:
    """Returns dictionary of paths to save files to download
    and urls to download file

    ex.
    {'C:\Home\Project\file.json':"https://website/file.json"}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """
    url_dict = {}
    for save_path, link in models_json_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
        json_filepath = save_path.replace("_fullmodel.h5", ".json")
        if not os.path.isfile(json_filepath):
            json_link = link.replace("_fullmodel.h5", ".json")
            url_dict[json_filepath] = json_link

    return url_dict

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

#### choose zenodo release
root = Tk()
choices = ['sat_RGB_2class_7384255', 'sat_5band_2class_7388008', 
            'sat_RGB_4class_6950472', 'sat_5band_4class_7344606', 
            'sat_NDWI_4class_7352859', 'sat_MNDWI_4class_7352850', 'sat_7band_4class_7358284',
            'aerial_2class_6234122', 'aerial_2class_6235090', 'ortho_2class_6410157']

variable = StringVar(root)
variable.set('sat_RGB_4class_6950472')
w = OptionMenu(root, variable, *choices)
w.pack(); root.mainloop()

dataset_id = variable.get()
print("Dataset ID : {}".format(dataset_id))

zenodo_id = dataset_id.split('_')[-1]
print("Zenodo ID : {}".format(zenodo_id))

## choose model implementation type
root = Tk()
choices = ['BEST','ENSEMBLE']
variable = StringVar(root)
variable.set('ENSEMBLE')
w = OptionMenu(root, variable, *choices)
w.pack(); root.mainloop()

# directory that holds all downloaded models
downloaded_models_dir = os.getcwd() + os.sep+ 'downloaded_models'
# directory that holds specific downloaded model
model_direc = os.path.join(downloaded_models_dir, dataset_id)

model_choice = variable.get()
print("Model implementation choice : {}".format(model_choice))

####======================================

if not os.path.isdir(downloaded_models_dir):
    print(f"Creating {downloaded_models_dir}")
    os.mkdir(downloaded_models_dir)


model_direc = os.path.join(downloaded_models_dir, dataset_id)
if not os.path.isdir(model_direc):
    print(f"Creating {model_direc}")
    os.mkdir(model_direc)

# Send request to zenodo for selected model by zenodo_id

root_url = 'https://zenodo.org/api/records/'+zenodo_id

r = requests.get(root_url)

# get list of all files associated with zenodo id
js = json.loads(r.text)
files = js['files']
zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
# if no zipped models exist then perform async download 
if zipped_model_list != []:
    print("Checking for zipped model")
    zip_url=zipped_model_list[0]['links']['self']
    # print(f'zip_url: {zip_url}')
    weights_direc = os.path.abspath(r'C:\1_USGS\5_Doodleverse\segmentation_zoo\segmentation_zoo\scripts\downloaded_models')
    # update weights_direc to directory of downloaded models
    weights_direc =download_zipped_model(dataset_id,weights_direc,zip_url)
    print(f"Weights directory: {weights_direc}")
# if no zipped models exist then perform async download 
elif zipped_model_list == []:

    # dictionary to hold urls and full path to save models at
    models_json_dict = {}

    if model_choice=='BEST':
        # retrieve best model text file
        best_model_json = [f for f in files if f["key"] == "BEST_MODEL.txt"][0]
        best_model_txt_path = model_direc + os.sep + "BEST_MODEL.txt"
        
        # if best BEST_MODEL.txt file not exist then download it
        if not os.path.isfile(best_model_txt_path):
            download_url(
                best_model_json["links"]["self"],
                best_model_txt_path,
            )
        
        # read in that file 
        with open(best_model_txt_path) as f:
            filename = f.read()

        # check if json and h5 file in BEST_MODEL.txt exist
        model_json = [f for f in files if f["key"] == filename][0]
        # path to save model
        outfile = model_direc + os.sep + filename
        
        # path to save file and json data associated with file saved to dict
        models_json_dict[outfile] = model_json["links"]["self"]
        url_dict = get_url_dict_to_download(models_json_dict)
        # if any files are not found locally download them asynchronous
        if url_dict != {}:
            run_async_download(url_dict)
        

        # get best model config and download it
        best_model_json = [f for f in files if f['key']==filename.replace('_fullmodel.h5','.json')]
        outfile = model_direc + os.sep + filename.replace('_fullmodel.h5','.json')
        if not os.path.isfile(outfile):
            print("\nDownloading file to {}".format(outfile))
            download_url(best_model_json[0]['links']['self'], outfile)

    else:
        # get list of all models
        all_models = [f for f in files if f["key"].endswith(".h5")]
        # print(f"\nModels available : {all_models }")
        # check if all h5 files in files are in model_direc
        for model_json in all_models:
            outfile = (
                model_direc
                + os.sep
                + model_json["links"]["self"].split("/")[-1]
            )
            # print(f"\nENSEMBLE: outfile: {outfile}")
            # path to save file and json data associated with file saved to dict
            models_json_dict[outfile] = model_json["links"]["self"]
        # print(f"\nmodels_json_dict: {models_json_dict}")
        url_dict = get_url_dict_to_download(models_json_dict)
        print(f"\nURLs to download: {url_dict}")
        # if any files are not found locally download them asynchronous
        if url_dict != {}:
            run_async_download(url_dict)
    
###==============================================

root = Tk()
root.filename =  filedialog.askdirectory(title = "Select directory of images (or npzs) to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

#####################################
#### concatenate models
####################################

# W : list containing all the weight files fill paths
W=[]

if model_choice=='ENSEMBLE':
    W= glob(model_direc+os.sep+'*.h5')
    print("{} sets of model weights were found ".format(len(W)))
else:
    #read best model file
    #select weights
    with open(model_direc+os.sep+'BEST_MODEL.txt') as f:
        w = f.readlines()
    W = [model_direc + os.sep + w[0]]


# For each set of weights in W load them in
M= []; C=[]; T = []
for counter,weights in enumerate(W):

    try:
        # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
        # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
        configfile = weights.replace('_fullmodel.h5','.json').replace('weights', 'config')
        with open(configfile) as f:
            config = json.load(f)
    except:
        # Turn the .h5 file into a json so that the data can be loaded into dynamic variables        
        configfile = weights.replace('.h5','.json').replace('weights', 'config')
        with open(configfile) as f:
            config = json.load(f)
    # Dynamically creates all variables from config dict.
    # For example configs's {'TARGET_SIZE': [768, 768]} will be created as TARGET_SIZE=[768, 768]
    # This is how the program is able to use variables that have never been explicitly defined
    for k in config.keys():
        exec(k+'=config["'+k+'"]')


    if counter==0:
        #####################################
        #### hardware
        ####################################

        SET_GPU = str(SET_GPU)

        if SET_GPU != '-1':
            USE_GPU = True
            print('Using GPU')
        else:
            USE_GPU = False
            print('Using CPU')

        if len(SET_GPU.split(','))>1:
            USE_MULTI_GPU = True 
            print('Using multiple GPUs')
        else:
            USE_MULTI_GPU = False
            if USE_GPU:
                print('Using single GPU device')
            else:
                print('Using single CPU device')

        #suppress tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if USE_GPU == True:
            os.environ['CUDA_VISIBLE_DEVICES'] = SET_GPU

            from doodleverse_utils.prediction_imports import *
            from tensorflow.python.client import device_lib
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            print(physical_devices)

            if physical_devices:
                # Restrict TensorFlow to only use the first GPU
                try:
                    tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
                except RuntimeError as e:
                    # Visible devices must be set at program startup
                    print(e)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            from doodleverse_utils.prediction_imports import *
            from tensorflow.python.client import device_lib
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            print(physical_devices)

        ### mixed precision
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        # tf.debugging.set_log_device_placement(True)

        for i in physical_devices:
            tf.config.experimental.set_memory_growth(i, True)
        print(tf.config.get_visible_devices())

        if USE_MULTI_GPU:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy([p.name.split('/physical_device:')[-1] for p in physical_devices], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            print("Number of distributed devices: {}".format(strategy.num_replicas_in_sync))


    #from imports import *
    from doodleverse_utils.imports import *
    from doodleverse_utils.model_imports import *

    #---------------------------------------------------

    #=======================================================
    # Import the architectures for following models from doodleverse_utils
    # 1. custom_resunet
    # 2. custom_unet
    # 3. simple_resunet
    # 4. simple_unet
    # 5. satunet
    # 6. custom_resunet
    # 7. custom_satunet

    # Get the selected model based on the weights file's MODEL key provided
    # create the model with the data loaded in from the weights file
    print('.....................................')
    print('Creating and compiling model {}...'.format(counter))

    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        )
    elif MODEL=='unet':
        model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        )

    elif MODEL =='simple_resunet':

        model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='simple_unet':
        model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='satunet':

        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    else:
        print("Model must be one of 'unet', 'resunet', or 'satunet'")
        sys.exit(2)

    try:
        # Load in the model from the weights which is the location of the weights file        
        model = tf.keras.models.load_model(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)
        
    except:
        # Load the metrics mean_iou, dice_coef from doodleverse_utils
        # Load in the custom loss function from doodleverse_utils        
        model.compile(optimizer = 'adam', loss = dice_coef_loss(NCLASSES))#, metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])

        model.load_weights(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)

# metadatadict contains the model name (T) the config file(C) and the model weights(W)
metadatadict = {}
metadatadict['model_weights'] = W
metadatadict['config_files'] = C
metadatadict['model_types'] = T

#####################################
#### read images
####################################

# The following lines prepare the data to be predicted
sample_filenames = sorted(glob(sample_direc+os.sep+'*.*'))
if sample_filenames[0].split('.')[-1]=='npz':
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.npz'))
else:
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
    if len(sample_filenames)==0:
        sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

#####################################
#### run model on each image in a for loop
####################################
### predict
print('.....................................')
print('Using model for prediction on images ...')

#look for TTA config
if not 'TESTTIMEAUG' in locals():
    print("TESTTIMEAUG not found in config file(s). Setting to False")
    TESTTIMEAUG = False

if not 'WRITE_MODELMETADATA' in locals():
    print("WRITE_MODELMETADATA not found in config file(s). Setting to False")
    WRITE_MODELMETADATA = False
if not 'OTSU_THRESHOLD' in locals():
    print("OTSU_THRESHOLD not found in config file(s). Setting to False")
    OTSU_THRESHOLD = False

# Import do_seg() from doodleverse_utils to perform the segmentation on the images
for f in auto_tqdm(sample_filenames):
    try:
        do_seg(f, M, metadatadict, sample_direc,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD)
    except:
        print("{} failed. Check config file, and check the path provided contains valid imagery".format(f))


