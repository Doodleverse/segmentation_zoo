# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022-2023, Marda Science LLC
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
import sys, os, json
import traceback

# local imports
import model_functions

# external imports
from tkinter import filedialog
from tkinter import *
import tensorflow as tf  
import tensorflow.keras.backend as K

# do_parallel = True 
do_parallel = False

# profile = 'full' ## predseg + meta +overlay
profile = 'meta' ## predseg + meta 
# profile = 'minimal' ## predseg

if do_parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
        
    #### choose generic task 
    root = Tk()
    choices = [
        "aerial_watermasking",
        "aerial_landcover",
        "satellite_shorelines",
        "generic_landcover_highres",
        "coastal_landcover_highres"
    ]

    variable = StringVar(root)
    variable.set("aerial_watermasking")
    w = OptionMenu(root, variable, *choices)
    w.pack()
    root.mainloop()

    task_id = variable.get()
    print("You chose task : {}".format(task_id))


    #### choose zenodo release
    root = Tk()

    if task_id=="aerial_watermasking":

        choices = [
        "noaa_oblique_2class_7604083",
        "aerial_oblique_2class_7604075",
        "aerial_nadir_2class_7604077",
        "pcmsc_wm_v2_7700430"
        ]

        variable = StringVar(root)
        variable.set("aerial_oblique_2class_7604075")

    elif task_id=="aerial_landcover":

        choices = [
        "floodnet_10class_7566810",
        "noaa_4class_7631354"
        ]

        variable = StringVar(root)
        variable.set("noaa_4class_7631354")

    elif task_id=="satellite_shorelines":

        choices = [
        "sat_RGB_2class_7448405",
        "sat_5band_2class_7448390",
        "sat_NDWI_2class_7557072",
        "sat_MNDWI_2class_7557080",
        "sat_RGB_4class_6950472",
        "sat_5band_4class_7344606",
        "sat_NDWI_4class_7352859",
        "sat_MNDWI_4class_7352850",
        "sat_7band_4class_7358284"
        ]

        variable = StringVar(root)
        variable.set("sat_RGB_4class_6950472")

    elif task_id=="generic_landcover_highres":

        choices = [
        "openearthmap_9class_7576894",
        "deepglobe_7class_7576898",
        "enviroatlas_6class_7576909",
        "aaai_building_7607895",
        "aaai_floodedbuildings_7622733",
        "xbd_building_7613212",
        "xbd_damagedbuilding_7613175"
        ]
        # "floodnet_10class_7566797", this is the 1024x768 px version

        variable = StringVar(root)
        variable.set("openearthmap_9class_7576894")

    elif task_id=="coastal_landcover_highres":

        choices = [
            "orthoCT_2class_7574784", 
            "orthoCT_5class_7566992",
            "orthoCT_5class_segformer_7641708",
            "orthoCT_8class_7570583",
            "orthoCT_8class_segformer_7641724",
            "chesapeake_7class_7576904",
            "chesapeake_7class_segformer_7677506"
        ]
        # add: barrierIslands

        variable = StringVar(root)
        variable.set("orthoCT_5class_7566992")

    #=============================
    w = OptionMenu(root, variable, *choices)
    w.pack()
    root.mainloop()

    dataset_id = variable.get()
    print("You chose dataset ID : {}".format(dataset_id))

    zenodo_id = dataset_id.split("_")[-1]
    print("Zenodo ID : {}".format(zenodo_id))

    ## choose model implementation type
    root = Tk()
    choices = ["BEST", "ENSEMBLE"]
    variable = StringVar(root)
    variable.set("BEST")
    w = OptionMenu(root, variable, *choices)
    w.pack()
    root.mainloop()

    model_choice = variable.get()
    print("Model implementation choice : {}".format(model_choice))

    ####======================================

    # segmentation zoo directory
    parent_direc = os.path.dirname(os.getcwd())
    # create downloaded models directory in segmentation_zoo/downloaded_models
    downloaded_models_dir = get_models_dir = model_functions.get_model_dir(parent_direc, "downloaded_models")
    print(f"Downloaded Models Located at: {downloaded_models_dir}")
    # directory to hold specific downloaded model
    model_direc = model_functions.get_model_dir(downloaded_models_dir, dataset_id)

    # get list of available files to download for zenodo id
    files = model_functions.request_available_files(zenodo_id)
    # print(f"Available files for zenodo {zenodo_id}: {files}")

    zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
    # check if zenodo release contains zip file 'rgb.zip'
    is_zip = model_functions.is_zipped_release(files)
    # zenodo release contained file 'rgb.zip' download it and unzip it
    if is_zip:
        print("Checking for zipped model")
        zip_url = zipped_model_list[0]["links"]["self"]
        model_direc = model_functions.download_zipped_model(model_direc, zip_url)
    # zenodo release contained no zip files. perform async download
    elif is_zip == False:
        if model_choice == "BEST":
            model_functions.download_BEST_model(files, model_direc)
        elif model_choice == "ENSEMBLE":
            model_functions.download_ENSEMBLE_model(files, model_direc)

    ###==============================================

    root = Tk()
    root.filename = filedialog.askdirectory(title="Select directory of images (or npzs) to segment")
    sample_direc = root.filename
    print(sample_direc)
    root.withdraw()

    #####################################
    #### concatenate models
    ####################################

    # weights_files : list containing all the weight files fill paths
    weights_files = model_functions.get_weights_list(model_choice, model_direc)

    # For each set of weights in weights_files load them in
    M = []
    C = []
    T = []
    for counter, weights in enumerate(weights_files):

        try:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            configfile = weights.replace("_fullmodel.h5", ".json").replace("weights", "config").strip()
            with open(configfile) as file:
                config = json.load(file)
        except:
            # Turn the .h5 file into a json so that the data can be loaded into dynamic variables
            configfile = weights.replace(".h5", ".json").replace("weights", "config").strip()
            with open(configfile) as file:
                config = json.load(file)
        # Dynamically creates all variables from config dict.
        # For example configs's {'TARGET_SIZE': [768, 768]} will be created as TARGET_SIZE=[768, 768]
        # This is how the program is able to use variables that have never been explicitly defined
        for k in config.keys():
            exec(k + '=config["' + k + '"]')

        print("Using CPU")
        if counter == 0:
            from doodleverse_utils.prediction_imports import *

            if MODEL!='segformer':
                ### mixed precision
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")

            print(tf.config.get_visible_devices())

        # Get the selected model based on the weights file's MODEL key provided
        # create the model with the data loaded in from the weights file
        print("Creating and compiling model {}...".format(counter))
        try:
            model, model_list, config_files, model_names = model_functions.get_model(weights_files)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'")
            sys.exit(2)

        # get dictionary containing all files needed to run models on data
        metadatadict = model_functions.get_metadatadict(weights_files, config_files, model_names)

    # read contents of config file into dictionary
    config = model_functions.get_config(weights_files)
    TARGET_SIZE = config.get("TARGET_SIZE")
    NCLASSES = config.get("NCLASSES")
    N_DATA_BANDS = config.get("N_DATA_BANDS")

    # metadatadict contains model names, config files, and, model weights(weights_files)
    metadatadict = {}
    metadatadict["model_weights"] = weights_files
    metadatadict["config_files"] = config_files
    metadatadict["model_types"] = model_names
    print(f"\n metadatadict:\n {metadatadict}")

    #####################################
    # read images
    #####################################

    sample_filenames = model_functions.sort_files(sample_direc)
    print("Number of samples: %i" % (len(sample_filenames)))

    #####################################
    #### run model on each image in a for loop
    ####################################
    print(".....................................")
    print("Using model for prediction on images ...")

    # look for TTA config
    if not "TESTTIMEAUG" in locals():
        print("TESTTIMEAUG not found in config file(s). Setting to False")
        TESTTIMEAUG = False

    if not "WRITE_MODELMETADATA" in locals():
        print("WRITE_MODELMETADATA not found in config file(s). Setting to False")
        WRITE_MODELMETADATA = True
    if not "OTSU_THRESHOLD" in locals():
        print("OTSU_THRESHOLD not found in config file(s). Setting to False")
        OTSU_THRESHOLD = False


    print(f"TESTTIMEAUG: {TESTTIMEAUG}")
    print(f"WRITE_MODELMETADATA: {WRITE_MODELMETADATA}")
    print(f"OTSU_THRESHOLD: {OTSU_THRESHOLD}")

    # run models on imagery
    try:
        print(f"file: {file}")
        model_functions.compute_segmentation(
            TARGET_SIZE,
            N_DATA_BANDS,
            NCLASSES,
            MODEL,
            sample_direc,
            model_list,
            metadatadict,
            do_parallel,
            profile
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print(f"{file} failed. Check config file, and check the path provided contains valid imagery")
