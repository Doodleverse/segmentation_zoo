import sys,os, json
import asyncio
import platform
from glob import glob

import tqdm
from tqdm.auto import tqdm as auto_tqdm
import tqdm.asyncio
import zipfile
from tkinter import filedialog
import requests
import aiohttp
import tensorflow as tf
# specify where imports come from
from doodleverse_utils.prediction_imports import do_seg
# Import the architectures for following models from doodleverse_utils
from doodleverse_utils.model_imports import (
    simple_resunet,
    custom_resunet,
    custom_unet,
    simple_unet,
    simple_resunet,
    simple_satunet,
)
from doodleverse_utils.model_imports import dice_coef_loss

def get_weights_list(model_choice: str,weights_direc:str):
        """Returns of the weights files(.h5) within weights_direc """
        if model_choice == 'ENSEMBLE':
            return glob(weights_direc + os.sep + '*.h5')
        elif model_choice == 'BEST':
            with open(weights_direc + os.sep + 'BEST_MODEL.txt')as f:
                w = f.readlines()
            return [weights_direc + os.sep + w[0]]

def get_metadatadict(
    weights_list: list, config_files: list, model_names: list
):
    metadatadict = {}
    metadatadict["model_weights"] = weights_list
    metadatadict["config_files"] = config_files
    metadatadict["model_names"] = model_names
    return metadatadict

def get_config(weights_list:list)->dict:
    weights_file = weights_list[0]
    configfile = weights_file.replace(".h5", ".json").replace("weights", "config")
    if "fullmodel" in configfile:
        configfile = configfile.replace("_fullmodel", "")
    with open(configfile) as f:
        config = json.load(f)
    return config

def get_model(weights_list: list):
        model_list = []
        config_files = []
        model_names = []
        if weights_list == []:
            raise Exception("No Model Info Passed")
        for weights in weights_list:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            configfile = weights.replace(".h5", ".json").replace("weights", "config")
            if "fullmodel" in configfile:
                configfile = configfile.replace("_fullmodel", "")
            with open(configfile) as f:
                config = json.load(f)
            TARGET_SIZE = config.get("TARGET_SIZE")
            MODEL = config.get("MODEL")
            NCLASSES = config.get("NCLASSES")
            KERNEL = config.get("KERNEL")
            STRIDE = config.get("STRIDE")
            FILTERS = config.get("FILTERS")
            N_DATA_BANDS = config.get("N_DATA_BANDS")
            DROPOUT = config.get("DROPOUT")
            DROPOUT_CHANGE_PER_LAYER = config.get("DROPOUT_CHANGE_PER_LAYER")
            DROPOUT_TYPE = config.get("DROPOUT_TYPE")
            USE_DROPOUT_ON_UPSAMPLING = config.get("USE_DROPOUT_ON_UPSAMPLING")
            DO_TRAIN = config.get("DO_TRAIN")
            LOSS = config.get("LOSS")
            PATIENCE = config.get("PATIENCE")
            MAX_EPOCHS = config.get("MAX_EPOCHS")
            VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")
            RAMPUP_EPOCHS = config.get("RAMPUP_EPOCHS")
            SUSTAIN_EPOCHS = config.get("SUSTAIN_EPOCHS")
            EXP_DECAY = config.get("EXP_DECAY")
            START_LR = config.get("START_LR")
            MIN_LR = config.get("MIN_LR")
            MAX_LR = config.get("MAX_LR")
            FILTER_VALUE = config.get("FILTER_VALUE")
            DOPLOT = config.get("DOPLOT")
            ROOT_STRING = config.get("ROOT_STRING")
            USEMASK = config.get("USEMASK")
            AUG_ROT = config.get("AUG_ROT")
            AUG_ZOOM = config.get("AUG_ZOOM")
            AUG_WIDTHSHIFT = config.get("AUG_WIDTHSHIFT")
            AUG_HEIGHTSHIFT = config.get("AUG_HEIGHTSHIFT")
            AUG_HFLIP = config.get("AUG_HFLIP")
            AUG_VFLIP = config.get("AUG_VFLIP")
            AUG_LOOPS = config.get("AUG_LOOPS")
            AUG_COPIES = config.get("AUG_COPIES")
            REMAP_CLASSES = config.get("REMAP_CLASSES")
            try:
                model = tf.keras.models.load_model(weights)
                #  nclasses=NCLASSES, may have to replace nclasses with NCLASSES
            except BaseException:
                if MODEL == "resunet":
                    model = custom_resunet(
                        (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            NCLASSES + 1 if NCLASSES == 1 else NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )
                elif MODEL == "unet":
                    model = custom_unet(
                        (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            NCLASSES + 1 if NCLASSES == 1 else NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )
                elif MODEL == "simple_resunet":
                    # num_filters = 8 # initial filters
                    model = simple_resunet(
                        (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            NCLASSES + 1 if NCLASSES == 1 else NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                # 346,564
                elif MODEL == "simple_unet":
                    model = simple_unet(
                        (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            NCLASSES + 1 if NCLASSES == 1 else NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL=='satunet':
                    model = simple_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
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
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model."
                    )
                # Load in the custom loss function from doodleverse_utils
                model.compile(
                    optimizer="adam", loss=dice_coef_loss(NCLASSES)
                )  # , metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])
                model.load_weights(weights)
            
            # try:
            #     # Load in the model from the weights which is the location of the weights file  
            #     print(f"Using the model: {model}") 
            #     model = tf.keras.models.load_model(weights)
            # except:
            #     # Load the metrics mean_iou, dice_coef from doodleverse_utils
            #     # Load in the custom loss function from doodleverse_utils 
            #     model.compile(
            #         optimizer="adam", loss=dice_coef_loss(NCLASSES)
            #     )  # , metrics = [iou_multi(NCLASSESNCLASSES), dice_multi(NCLASSESNCLASSES)])
            #     model.load_weights(weights)

            model_names.append(MODEL)
            model_list.append(model)
            config_files.append(configfile)

        return model, model_list, config_files, model_names
 
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
    
def compute_segmentation(
    TARGET_SIZE:tuple,
    N_DATA_BANDS:int,
    NCLASSES:int,
    sample_direc: str,
    model_list: list,
    metadatadict: dict,
):
    # look for TTA config
    if "TESTTIMEAUG" not in locals():
        TESTTIMEAUG = False
    WRITE_MODELMETADATA = False
    # Read in the image filenames as either .npz,.jpg, or .png
    files_to_segment = sort_files(sample_direc)
    # Compute the segmentation for each of the files
    for file_to_seg in tqdm.auto.tqdm(files_to_segment):
        do_seg(
            file_to_seg,
            model_list,
            metadatadict,
            sample_direc=sample_direc,
            NCLASSES=NCLASSES,
            N_DATA_BANDS=N_DATA_BANDS,
            TARGET_SIZE=TARGET_SIZE,
            TESTTIMEAUG=TESTTIMEAUG,
            WRITE_MODELMETADATA=WRITE_MODELMETADATA,
            OTSU_THRESHOLD=False,
        )