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
# from doodleverse_utils.prediction_imports import do_seg

import os  # , time

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# from skimage.filters.rank import median
# from skimage.morphology import disk
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import json
from skimage.io import imsave, imread
from numpy.lib.stride_tricks import as_strided as ast
from glob import glob

# from joblib import Parallel, delayed
# from skimage.morphology import remove_small_holes, remove_small_objects
# from scipy.ndimage import maximum_filter
from skimage.transform import resize

# from tqdm import tqdm
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

import tensorflow as tf  # numerical operations on gpu
import tensorflow.keras.backend as K

def seg_file2tensor_3band(f, TARGET_SIZE):  # , resize):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    bigimage = imread(f)  # Image.open(f)
    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    # smallimage=bigimage.resize((TARGET_SIZE[1], TARGET_SIZE[0]))
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

def seg_file2tensor_ND(f, TARGET_SIZE):  # , resize):
    """
    "seg_file2tensor(f)"
    This function reads a NPZ image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of npz
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    with np.load(f) as data:
        bigimage = data["arr_0"].astype("uint8")

    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    # smallimage=bigimage.resize((TARGET_SIZE[1], TARGET_SIZE[0]))
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

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

def do_seg1(
    f, M, metadatadict, sample_direc, 
    NCLASSES, N_DATA_BANDS, TARGET_SIZE, TESTTIMEAUG, WRITE_MODELMETADATA,
    OTSU_THRESHOLD
):

    if f.endswith("jpg"):
        segfile = f.replace(".jpg", "_predseg.png")
    elif f.endswith("png"):
        segfile = f.replace(".png", "_predseg.png")
    elif f.endswith("npz"):  # in f:
        segfile = f.replace(".npz", "_predseg.png")

    if WRITE_MODELMETADATA:
        metadatadict["input_file"] = f

    segfile = os.path.normpath(segfile)
    segfile = segfile.replace(
        os.path.normpath(sample_direc), os.path.normpath(sample_direc + os.sep + "out")
    )

    try:
        os.mkdir(os.path.normpath(sample_direc + os.sep + "out"))
    except:
        pass

    if WRITE_MODELMETADATA:
        metadatadict["nclasses"] = NCLASSES
        metadatadict["n_data_bands"] = N_DATA_BANDS

    if NCLASSES == 2:

        if N_DATA_BANDS <= 3:
            image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
        else:
            image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

        image = standardize(image.numpy()).squeeze()

        E0 = []
        E1 = []

        for counter, model in enumerate(M):
            # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)

            try:
                est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).squeeze()
            except:
                est_label = model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1).squeeze()

            if TESTTIMEAUG == True:
                # return the flipped prediction
                est_label2 = np.flipud(
                    model.predict(
                        tf.expand_dims(np.flipud(image), 0), batch_size=1
                    ).squeeze()
                )
                est_label3 = np.fliplr(
                    model.predict(
                        tf.expand_dims(np.fliplr(image), 0), batch_size=1
                    ).squeeze()
                )
                est_label4 = np.flipud(
                    np.fliplr(
                        model.predict(
                            tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1
                        ).squeeze()
                    )
                )

                # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
                est_label = est_label + est_label2 + est_label3 + est_label4
                del est_label2, est_label3, est_label4

            E0.append(
                resize(est_label[:, :, 0], (w, h), preserve_range=True, clip=True)
            )
            E1.append(
                resize(est_label[:, :, 1], (w, h), preserve_range=True, clip=True)
            )
            del est_label

        # heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
        K.clear_session()

        e0 = np.average(np.dstack(E0), axis=-1)  # , weights=np.array(MW))

        del E0

        e1 = np.average(np.dstack(E1), axis=-1)  # , weights=np.array(MW))
        del E1

        est_label = (e1 + (1 - e0)) / 2

        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = np.dstack((e0,e1))
        del e0, e1

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        # if DO_CRF:
        #     est_label, l_unique = crf_refine(softmax_scores, bigimage, NCLASSES+1, 1, 1, 2)

        #     est_label = est_label-1
        #     if WRITE_MODELMETADATA:
        #         metadatadict["otsu_threshold"] = np.nan

        if OTSU_THRESHOLD:
            thres = threshold_otsu(est_label)
            # print("Class threshold: %f" % (thres))
            est_label = (est_label > thres).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = thres

        else:
            # print("Not using Otsu threshold")
            est_label = (est_label > 0.5).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = 0.5            

    else:  ###NCLASSES>2

        if N_DATA_BANDS <= 3:
            image, w, h, bigimage = seg_file2tensor_3band(
                f, TARGET_SIZE
            )  # , resize=True)
            w = w.numpy()
            h = h.numpy()
        else:
            image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

        # image = tf.image.per_image_standardization(image)
        image = standardize(image.numpy())
        # return the base prediction
        if N_DATA_BANDS == 1:
            image = image[:, :, 0]
            bigimage = np.dstack((bigimage, bigimage, bigimage))

        est_label = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], NCLASSES))
        for counter, model in enumerate(M):
            # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)

            est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).squeeze()

            if TESTTIMEAUG == True:
                # return the flipped prediction
                est_label2 = np.flipud(
                    model.predict(
                        tf.expand_dims(np.flipud(image), 0), batch_size=1
                    ).squeeze()
                )
                est_label3 = np.fliplr(
                    model.predict(
                        tf.expand_dims(np.fliplr(image), 0), batch_size=1
                    ).squeeze()
                )
                est_label4 = np.flipud(
                    np.fliplr(
                        model.predict(
                            tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1
                        ).squeeze()
                    )
                )

                # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
                est_label = est_label + est_label2 + est_label3 + est_label4
                del est_label2, est_label3, est_label4

            K.clear_session()

        est_label /= counter + 1
        print(f"type: est_label: {type(est_label)}")
        print(f"est_label: {est_label}")
        est_label = est_label.astype('float32')
        est_label = resize(est_label, (w, h))
        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = est_label.copy() #np.dstack((e0,e1))

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        # if DO_CRF:
        #     est_label, l_unique = crf_refine(softmax_scores, bigimage, NCLASSES, 1, 1, 2)

        #     est_label = est_label-1
        #     if WRITE_MODELMETADATA:
        #         metadatadict["otsu_threshold"] = np.nan

        # else:
        est_label = np.argmax(softmax_scores, -1)

    # heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)

    class_label_colormap = [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
    ]
    # add classes for more than 10 classes

    # if NCLASSES > 1:
    class_label_colormap = class_label_colormap[:NCLASSES]
    # else:
    #     class_label_colormap = class_label_colormap[:2]

    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    try:
        color_label = label_to_colors(
            est_label,
            bigimage.numpy()[:, :, 0] == 0,
            alpha=128,
            colormap=class_label_colormap,
            color_class_offset=0,
            do_alpha=False,
        )
    except:
        try:
            color_label = label_to_colors(
                est_label,
                bigimage[:, :, 0] == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )
        except:
            color_label = label_to_colors(
                est_label,
                bigimage == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )        

    imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)
    
    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    segfile = segfile.replace("_predseg.png", "_res.npz")

    if WRITE_MODELMETADATA:
        metadatadict["grey_label"] = est_label

        np.savez_compressed(segfile, **metadatadict)

    segfile = segfile.replace("_res.npz", "_overlay.png")

    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])

    plt.imshow(color_label, alpha=0.5)
    plt.axis("off")
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches="tight")
    plt.close("all")

    #### image - overlay side by side
    segfile = segfile.replace("_res.npz", "_image_overlay.png")

    plt.subplot(121)
    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])
    plt.axis("off")

    plt.subplot(122)
    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])
    if NCLASSES>2:
        plt.imshow(color_label, alpha=0.5)
    elif NCLASSES==2:
        cs = plt.contour(est_label, [-99,0,99], colors='r')
    plt.axis("off")
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches="tight")
    plt.close("all")

    if NCLASSES==2:
        segfile = segfile.replace("_overlay.png", "_result.mat")
        p = cs.collections[0].get_paths()[0]
        v = p.vertices
        x = v[:,0]
        y = v[:,1]
        io.savemat(segfile, dict(x=x, y=y))


def sort_files(sample_direc:str)->list:
    """returns list of sorted filenames in sample_direc

    Args:
        sample_direc (str): full path to directory of imagery/npz
        to run models on

    Returns:
        list: list of sorted filenames in sample_direc
    """    
    # prepares data to be predicted
    sample_filenames = sorted(glob(sample_direc+os.sep+'*.*'))
    if sample_filenames[0].split('.')[-1]=='npz':
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.npz'))
    else:
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
        if len(sample_filenames)==0:
            sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))
    return sample_filenames

def get_weights_list(model_choice: str,weights_direc:str):
        """Returns of the weights files(.h5) within weights_direc """
        if model_choice == 'ENSEMBLE':
            return glob(weights_direc + os.sep + '*.h5')
        elif model_choice == 'BEST':
            with open(weights_direc + os.sep + 'BEST_MODEL.txt')as f:
                w = f.readlines()
            return [weights_direc + os.sep + w[0]]

def download_ENSEMBLE_model(files:list,model_direc:str)->None:
    """downloads all models from zenodo.

    Args:
        files (list): list of available files for zenodo release
        model_direc (str): directory of model to download
    """
    # dictionary to hold urls and full path to save models at
    models_json_dict = {}
    # list of all models
    all_models = [f for f in files if f["key"].endswith(".h5")]
    # check if all h5 files in files are in model_direc
    for model_json in all_models:
        outfile = (
            model_direc
            + os.sep
            + model_json["links"]["self"].split("/")[-1]
        )
        # path to save file and json data associated with file saved to dict
        models_json_dict[outfile] = model_json["links"]["self"]
    url_dict = get_url_dict_to_download(models_json_dict)
    print(f"\nURLs to download: {url_dict}")
    # if any files are not found locally download them asynchronous
    if url_dict != {}:
        run_async_download(url_dict)

def download_BEST_model(files:list,model_direc:str)->None:
    """downloads best model from zenodo.

    Args:
        files (list): list of available files for zenodo release
        model_direc (str): directory of model to download

    Raises:
        FileNotFoundError: if model filename in 'BEST_MODEL.txt' does not
        exist online
    """
    # dictionary to hold urls and full path to save models at    
    models_json_dict = {}
    # retrieve best model text file
    best_model_json = [f for f in files if f["key"] == "BEST_MODEL.txt"][0]
    best_model_txt_path = os.path.join(model_direc,"BEST_MODEL.txt")
    # if BEST_MODEL.txt file not exist download it
    if not os.path.isfile(best_model_txt_path):
        download_url(
            best_model_json["links"]["self"],
            best_model_txt_path,
        )
    
    # read in BEST_MODEL.txt file 
    with open(best_model_txt_path) as f:
        best_model_filename = f.read()
    print(f"Best Model filename: {best_model_filename}")
    # check if json and h5 file in BEST_MODEL.txt exist
    model_json = [f for f in files if f["key"] == best_model_filename]
    if model_json == []:
        FILE_NOT_ONLINE_ERROR= f"File {best_model_filename} not found online. Raise an issue on Github"
        raise FileNotFoundError(FILE_NOT_ONLINE_ERROR)
    # path to save model
    outfile = os.path.join(model_direc,best_model_filename)
    # path to save file and json data associated with file saved to dict
    models_json_dict[outfile] = model_json[0]["links"]["self"]
    url_dict = get_url_dict_to_download(models_json_dict)
    # if any files are not found locally download them asynchronous
    if url_dict != {}:
        run_async_download(url_dict) 
    
def download_zipped_model(model_direc:str,url:str):
    # 'rgb.zip' is name of directory containing model online
    filename='rgb'
    # outfile: full path to directory containing model files
    # example: 'c:/model_name/rgb'
    outfile = model_direc + os.sep + filename
    if os.path.exists(outfile):   
        print(f'\n Found model weights directory: {os.path.abspath(outfile)}')
    # if model directory does not exist download zipped model from Zenodo
    if not os.path.exists(outfile):
        print(f'\n Downloading to model weights directory: {os.path.abspath(outfile)}')
        zip_file=filename+'.zip'
        zip_folder = model_direc + os.sep + zip_file
        print(f'Retrieving model {url} ...')
        download_zip(url, zip_folder)
        print(f'Unzipping model to {model_direc} ...')
        with zipfile.ZipFile(zip_folder, 'r') as zip_ref:
            zip_ref.extractall(model_direc)
        print(f'Removing {zip_folder}')
        os.remove(zip_folder)
    
    # set weights dir to sub directory (rgb) containing model files  
    model_direc = os.path.join(model_direc,'rgb')
                
    # Ensure all files are unzipped
    with os.scandir(model_direc) as it:
        for entry in it:
            if entry.name.endswith('.zip'):
                with zipfile.ZipFile(entry, 'r') as zip_ref:
                    zip_ref.extractall(model_direc)
                os.remove(entry)
    return model_direc
            
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
    # chunk_size: int = 128
    # chunk_size: int = 1024
    chunk_size: int = 2048
    # raise a timeout 600 seconds (10 mins)
    async with session.get(url, raise_for_status=True,timeout=600) as r:
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

def get_model_dir(parent_directory:str, dir_name:str)->str:
    """returns full path to directory named dir_name and if it doesn't exist
    creates new directory dir_name within parent directory

    Args:
        parent_directory (str): directory to create new directory dir_name within
        dir_name (str): name of directory to get full path to

    Returns:
        str: full path to dir_name directory
    """    
    new_dir = os.path.join(parent_directory,dir_name)
    if not os.path.isdir(new_dir):
        print(f"Creating {new_dir}")
        os.mkdir(new_dir)
    return new_dir

def request_available_files(zenodo_id:str)->list:
    """returns list of available downloadable files for zenodo_id

    Args:
        zenodo_id (str): id of zenodo release

    Returns:
        list: list of available files downloadable for zenodo_id
    """    
    # Send request to zenodo for selected model by zenodo_id
    root_url = 'https://zenodo.org/api/records/'+zenodo_id
    r = requests.get(root_url)
    # get list of all files associated with zenodo id
    js = json.loads(r.text)
    files = js['files']
    return files

def is_zipped_release(files:list)->bool:
    """returns True if zenodo id contains 'rgb.zip' file otherwise returns false

    Args:
        files (list): list of available files for download

    Returns:
        bool: returns True if zenodo id contains 'rgb.zip' file otherwise returns false
    """    
    zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
    if zipped_model_list == []:
        return False
    return True

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

model_choice = variable.get()
print("Model implementation choice : {}".format(model_choice))

####======================================

# segmentation zoo directory
parent_direc=os.path.dirname(os.getcwd())
# create downloaded models directory in segmentation_zoo/downloaded_models
downloaded_models_dir = get_models_dir=get_model_dir(parent_direc,"downloaded_models")
print(f'Downloaded Models Located at: {downloaded_models_dir}')
# directory to hold specific downloaded model
model_direc = get_model_dir(downloaded_models_dir, dataset_id)

# get list of available files to download for zenodo id
files = request_available_files(zenodo_id)
print(f"Available files for zenodo {zenodo_id}: {files}")

zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
# check if zenodo release contains zip file 'rgb.zip'
is_zip = is_zipped_release(files)
# zenodo release contained file 'rgb.zip' download it and unzip it
if is_zip:
    print("Checking for zipped model")
    zip_url=zipped_model_list[0]['links']['self']
    model_direc = download_zipped_model(model_direc,zip_url)
# zenodo release contained no zip files. perform async download 
elif is_zip == False:
    if model_choice=='BEST':
        download_BEST_model(files,model_direc)
    elif model_choice=='ENSEMBLE':
        download_ENSEMBLE_model(files,model_direc)
    
###==============================================

root = Tk()
root.filename =  filedialog.askdirectory(title = "Select directory of images (or npzs) to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

#####################################
#### concatenate models
####################################

# weights_files : list containing all the weight files fill paths
weights_files = get_weights_list(model_choice,model_direc)

# For each set of weights in weights_files load them in
M= []; C=[]; T = []
for counter,weights in enumerate(weights_files):

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
        #@todo remove this
        SET_GPU = -1
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
        print(f"Using the model: {model}")      
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

# metadatadict contains  model name (T), config file(C) and, model weights(weights_files)
metadatadict = {}
metadatadict['model_weights'] = weights_files
metadatadict['config_files'] = C
metadatadict['model_types'] = T
print(f"\n metadatadict:\n {metadatadict}")
#####################################
# read images
#####################################

sample_filenames = sort_files(sample_direc)
print('Number of samples: %i' % (len(sample_filenames)))

#####################################
#### run model on each image in a for loop
####################################
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

print("Do seg")
print(f"sample_direc: {sample_direc}")
print(f"M: {M}")
print(f"NCLASSES: {NCLASSES}")
print(f"N_DATA_BANDS: {N_DATA_BANDS}")
print(f"TARGET_SIZE: {TARGET_SIZE}")
print(f"TESTTIMEAUG: {TESTTIMEAUG}")
#@todo remove this... just for testing
WRITE_MODELMETADATA = False
OTSU_THRESHOLD = False
print(f"WRITE_MODELMETADATA: {WRITE_MODELMETADATA}")
print(f"OTSU_THRESHOLD: {OTSU_THRESHOLD}")
import traceback
# Import do_seg() from doodleverse_utils to perform the segmentation on the images
for f in auto_tqdm(sample_filenames):
    try:
        print(f"file: {f}")
        do_seg1(f,
               M,
               metadatadict,
               sample_direc,
               NCLASSES,
               N_DATA_BANDS,
               TARGET_SIZE,
               TESTTIMEAUG, 
               WRITE_MODELMETADATA,
               OTSU_THRESHOLD)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("{} failed. Check config file, and check the path provided contains valid imagery".format(f))


