

import tensorflow as tf

from transformers import TFSegformerForSemanticSegmentation
import tensorflow.keras.backend as K

import numpy as np
import os, json, tqdm
from glob import glob
from doodleverse_utils.prediction_imports import seg_file2tensor_3band, standardize, resize, seg_file2tensor_ND
from doodleverse_utils.imports import label_to_colors, imsave

# Import the architectures for following models from doodleverse_utils
from doodleverse_utils.model_imports import (
    simple_resunet,
    custom_resunet,
    custom_unet,
    simple_unet,
    simple_resunet,
    simple_satunet,
    segformer
)


def sort_files(sample_direc: str) -> list:
    """returns list of sorted filenames in sample_direc

    Args:
        sample_direc (str): full path to directory of imagery/npz
        to run models on

    Returns:
        list: list of sorted filenames in sample_direc
    """
    # prepares data to be predicted
    sample_filenames = sorted(glob(sample_direc + os.sep + "*.*"))
    if sample_filenames[0].split(".")[-1] == "npz":
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.npz"))
    else:
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.jpg"))+sorted(glob(sample_direc + os.sep + "*.png"))+sorted(glob(sample_direc + os.sep + "*.tif"))
        
    return sample_filenames

# ===========

def get_model(weights_list: list):
    """Loads models in from weights list and loads in corresponding config file
    for each model weights file(.h5) in weights_list

    Args:
        weights_list (list): full path to model weights files(.h5)

    Raises:
        Exception: raised if weights_list is empty
        Exception: An unknown model type was loaded from any of weights files in
        weights_list

    Returns:
       model, model_list, config_files, model_names
    """
    model_list = []
    config_files = []
    model_names = []
    if weights_list == []:
        raise Exception("No Model Info Passed")
    for weights in weights_list:
        # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
        # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
        configfile = weights.replace(".h5", ".json").replace("weights", "config").strip()
        if "fullmodel" in configfile:
            configfile = configfile.replace("_fullmodel", "").strip()
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
            # Get the selected model based on the weights file's MODEL key provided
            # create the model with the data loaded in from the weights file
            # Load in the model from the weights which is the location of the weights file
            model = tf.keras.models.load_model(weights)
        except BaseException:
            if MODEL == "resunet":
                model = custom_resunet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES,
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
                    nclasses=NCLASSES,
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
                    nclasses=NCLASSES,
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
                    nclasses=NCLASSES,
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
            elif MODEL == "satunet":
                model = simple_satunet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    num_classes=NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1, 1),
                )
            elif MODEL=='segformer':
                id2label = {}
                for k in range(NCLASSES):
                    id2label[k]=str(k)
                model = segformer(id2label,num_classes=NCLASSES)
                # model.compile(optimizer='adam')
            else:
                raise Exception(f"An unknown model type {MODEL} was received. Please select a valid model.")
            # Load in custom loss function from doodleverse_utils
            # Load metrics mean_iou, dice_coef from doodleverse_utils
            # if MODEL!='segformer':
            #     model.compile(
            #         optimizer="adam", loss=dice_coef_loss(NCLASSES)
            #     )  # , metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])
            weights=weights.strip()
            model.load_weights(weights)

        model_names.append(MODEL)
        model_list.append(model)
        config_files.append(configfile)
    return model, model_list, config_files, model_names



def get_weights_list(model_choice: str, weights_direc: str) -> list:
    """Returns of list of full paths to weights files(.h5) within weights_direc

    Args:
        model_choice (str): 'ENSEMBLE' or 'BEST'
        weights_direc (str): full path to directory containing model weights

    Returns:
        list: list of full paths to weights files(.h5) within weights_direc
    """
    if model_choice == "ENSEMBLE":
        return glob(weights_direc + os.sep + "*.h5")
    elif model_choice == "BEST":
        with open(weights_direc + os.sep + "BEST_MODEL.txt") as f:
            w = f.readlines()
        return [weights_direc + os.sep + w[0]]


def get_config(weights_list: list) -> dict:
    """loads contents of config json files
    that have same name of h5 files in weights_list

    Args:
        weights_list (list): weight files(.h5) in weights_list

    Returns:
        dict: contents of config json files that have same name of h5 files in weights_list
    """
    weights_file = weights_list[0]
    configfile = weights_file.replace(".h5", ".json").replace("weights", "config").strip()
    if "fullmodel" in configfile:
        configfile = configfile.replace("_fullmodel", "").strip()
    with open(configfile.strip()) as f:
        config = json.load(f)
    return config

def get_metadatadict(weights_list: list, config_files: list, model_names: list) -> dict:
    """returns dictionary of model weights,config_files, and model_names

    Args:
        weights_list (list): list of full paths to weights files(.h5)
        config_files (list): list of full paths to config files(.json)
        model_names (list): list of model names

    Returns:
        dict: dictionary of model weights,config_files, and model_names
    """
    metadatadict = {}
    metadatadict["model_weights"] = weights_list
    metadatadict["config_files"] = config_files
    metadatadict["model_names"] = model_names
    return metadatadict

# #-----------------------------------
def get_image(f,N_DATA_BANDS,TARGET_SIZE,MODEL):
    if N_DATA_BANDS <= 3:
        image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
    else:
        image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

    try: ##>3 bands
        if N_DATA_BANDS<=3:
            if image.shape[-1]>3:
                image = image[:,:,:3]

            if bigimage.shape[-1]>3:
                bigimage = bigimage[:,:,:3]
    except:
        pass

    # print(f)
    # print(image.shape)

    image = standardize(image.numpy()).squeeze()

    if MODEL=='segformer':
        if np.ndim(image)==2:
            image = np.dstack((image, image, image))
        image = tf.transpose(image, (2, 0, 1))

    return image, w, h, bigimage 


# #-----------------------------------
def est_label_multiclass(image,M,MODEL,TESTTIMEAUG,NCLASSES,TARGET_SIZE):

    est_label = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], NCLASSES))
    
    for counter, model in enumerate(M):
        # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)
        try:
            if MODEL=='segformer':
                est_label = model(tf.expand_dims(image, 0)).logits
            else:
                est_label = tf.squeeze(model(tf.expand_dims(image, 0)))
        except:
            if MODEL=='segformer':
                #### FIX :3
                # est_label = model(tf.expand_dims(image[:,:,0], 0)).logits
                est_label = model(tf.expand_dims(image[:,:,:3], 0)).logits
            else:
                #### FIX :3                
                est_label = tf.squeeze(model(tf.expand_dims(image[:,:,0], 0)))

        if TESTTIMEAUG == True:
            # return the flipped prediction
            if MODEL=='segformer':
                est_label2 = np.flipud(
                    model(tf.expand_dims(np.flipud(image), 0)).logits
                    )                
            else:
                est_label2 = np.flipud(
                    tf.squeeze(model(tf.expand_dims(np.flipud(image), 0)))
                    )
            if MODEL=='segformer':

                est_label3 = np.fliplr(
                    model(
                        tf.expand_dims(np.fliplr(image), 0)).logits
                        )                
            else:
                est_label3 = np.fliplr(
                    tf.squeeze(model(tf.expand_dims(np.fliplr(image), 0)))
                )                
            if MODEL=='segformer':
                est_label4 = np.flipud(
                    np.fliplr(
                        tf.squeeze(model(tf.expand_dims(np.flipud(np.fliplr(image)), 0)).logits))
                )                
            else:
                est_label4 = np.flipud(
                    np.fliplr(
                        tf.squeeze(model(
                            tf.expand_dims(np.flipud(np.fliplr(image)), 0)))
                            ))
                
            # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
            est_label = est_label + est_label2 + est_label3 + est_label4

        K.clear_session()

    # heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
    return est_label, counter


# #-----------------------------------
def est_label_binary(image,M,MODEL,TESTTIMEAUG,NCLASSES,TARGET_SIZE,w,h):

    E0 = []
    E1 = []

    for counter, model in enumerate(M):
        # heatmap = make_gradcam_heatmap(tf.expand_dims(image, 0) , model)
        try:
            if MODEL=='segformer':
                # est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).logits
                est_label = model(tf.expand_dims(image, 0)).logits
            else:
                est_label = tf.squeeze(model.predict(tf.expand_dims(image, 0), batch_size=1))

        except:
            if MODEL=='segformer':
                #### FIX :3
                # est_label = model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1).logits
                est_label = model.predict(tf.expand_dims(image[:,:,:3], 0), batch_size=1).logits

            else:
                #### FIX :3
                est_label = tf.squeeze(model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1))

        if TESTTIMEAUG == True:
            # return the flipped prediction
            if MODEL=='segformer':
                est_label2 = np.flipud(
                    model.predict(tf.expand_dims(np.flipud(image), 0), batch_size=1).logits
                    )
            else:
                est_label2 = np.flipud(
                    tf.squeeze(model.predict(tf.expand_dims(np.flipud(image), 0), batch_size=1))
                    )

            if MODEL=='segformer':
                est_label3 = np.fliplr(
                    model.predict(
                        tf.expand_dims(np.fliplr(image), 0), batch_size=1).logits
                        )
            else:
                est_label3 = np.fliplr(
                    tf.squeeze(model.predict(
                        tf.expand_dims(np.fliplr(image), 0), batch_size=1))
                        )
                
            if MODEL=='segformer':
                est_label4 = np.flipud(
                    np.fliplr(
                        model.predict(
                            tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1).logits)
                            )
            else:
                est_label4 = np.flipud(
                    np.fliplr(
                        tf.squeeze(model.predict(
                            tf.expand_dims(np.flipud(np.fliplr(image)), 0), batch_size=1)))
                            )
                
            # soft voting - sum the softmax scores to return the new TTA estimated softmax scores
            est_label = est_label + est_label2 + est_label3 + est_label4
            # del est_label2, est_label3, est_label4
        
        est_label = est_label.numpy().astype('float32')

        if MODEL=='segformer':
            est_label = resize(est_label, (1, NCLASSES, TARGET_SIZE[0],TARGET_SIZE[1]), preserve_range=True, clip=True).squeeze()
            est_label = np.transpose(est_label, (1,2,0))

        E0.append(
            resize(est_label[:, :, 0], (w, h), preserve_range=True, clip=True)
        )
        E1.append(
            resize(est_label[:, :, 1], (w, h), preserve_range=True, clip=True)
        )
        # del est_label
    # heatmap = resize(heatmap,(w,h), preserve_range=True, clip=True)
    K.clear_session()

    return E0, E1 

# =========================================================
def do_seg(
    f, M, metadatadict, MODEL, sample_direc, 
    NCLASSES, N_DATA_BANDS, TARGET_SIZE, TESTTIMEAUG, WRITE_MODELMETADATA,
    OTSU_THRESHOLD,
    out_dir_name='out',
    profile='minimal'
):
    
    if profile=='meta':
        WRITE_MODELMETADATA = True
    if profile=='full':
        WRITE_MODELMETADATA = True

    # Mc = compile_models(M, MODEL)

    if f.endswith("jpg"):
        segfile = f.replace(".jpg", "_predseg.png")
    elif f.endswith("png"):
        segfile = f.replace(".png", "_predseg.png")
    elif f.endswith("tif"):
        segfile = f.replace(".tif", "_predseg.png")        
    elif f.endswith("npz"):  # in f:
        segfile = f.replace(".npz", "_predseg.png")

    if WRITE_MODELMETADATA:
        metadatadict["input_file"] = f
        
    # directory to hold the outputs of the models is named 'out' by default
    # create a directory to hold the outputs of the models, by default name it 'out' or the model name if it exists in metadatadict
    out_dir_path = os.path.normpath(sample_direc + os.sep + out_dir_name)
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    segfile = os.path.normpath(segfile)
    segfile = segfile.replace(
        os.path.normpath(sample_direc), os.path.normpath(sample_direc + os.sep + out_dir_name)
    )

    if WRITE_MODELMETADATA:
        metadatadict["nclasses"] = NCLASSES
        metadatadict["n_data_bands"] = N_DATA_BANDS

    if NCLASSES == 2:

        image, w, h, bigimage = get_image(f,N_DATA_BANDS,TARGET_SIZE,MODEL)

        if np.std(image)==0:

            print("Image {} is empty".format(f))
            e0 = np.zeros((w,h))
            e1 = np.zeros((w,h))

        else:

            E0, E1 = est_label_binary(image,M,MODEL,TESTTIMEAUG,NCLASSES,TARGET_SIZE,w,h)

            e0 = np.average(np.dstack(E0), axis=-1)  

            # del E0

            e1 = np.average(np.dstack(E1), axis=-1) 
            # del E1

        est_label = (e1 + (1 - e0)) / 2

        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = np.dstack((e0,e1))
        # del e0, e1

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        if OTSU_THRESHOLD:
            thres = threshold_otsu(est_label)
            # print("Class threshold: %f" % (thres))
            est_label = (est_label > thres).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = thres

        else:
            est_label = (est_label > 0.5).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = 0.5            

    else:  ###NCLASSES>2

        image, w, h, bigimage = get_image(f,N_DATA_BANDS,TARGET_SIZE,MODEL)

        if np.std(image)==0:

            print("Image {} is empty".format(f))
            est_label = np.zeros((w,h))

        else:
                
            est_label, counter = est_label_multiclass(image,M,MODEL,TESTTIMEAUG,NCLASSES,TARGET_SIZE)

            est_label /= counter + 1
            # est_label cannot be float16 so convert to float32
            est_label = est_label.numpy().astype('float32')

            if MODEL=='segformer':
                est_label = resize(est_label, (1, NCLASSES, TARGET_SIZE[0],TARGET_SIZE[1]), preserve_range=True, clip=True).squeeze()
                est_label = np.transpose(est_label, (1,2,0))
                est_label = resize(est_label, (w, h))
            else:
                est_label = resize(est_label, (w, h))


        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = est_label.copy() #np.dstack((e0,e1))

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        if np.std(image)>0:
            est_label = np.argmax(softmax_scores, -1)
        else:
            est_label = est_label.astype('uint8')


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
        "#ffe4e1",
        "#ff7373",
        "#666666",
        "#c0c0c0",
        "#66cdaa",
        "#afeeee",
        "#0e2f44",
        "#420420",
        "#794044",
        "#3399ff",
    ]

    class_label_colormap = class_label_colormap[:NCLASSES]

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

    if profile == 'full': #(profile !='minimal') and (profile !='meta'):
        #### plot overlay
        segfile = segfile.replace("_res.npz", "_overlay.png")

        if N_DATA_BANDS <= 3:
            plt.imshow(bigimage, cmap='gray')
        else:
            plt.imshow(bigimage[:, :, :3])

        plt.imshow(color_label, alpha=0.5)
        plt.axis("off")
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
        plt.imshow(color_label, alpha=0.5)
        plt.axis("off")
        plt.savefig(segfile, dpi=200, bbox_inches="tight")
        plt.close("all")

    if profile == 'full': #(profile !='minimal') and (profile !='meta'):

        #### plot overlay of per-class probabilities
        for kclass in range(softmax_scores.shape[-1]):
            tmpfile = segfile.replace("_overlay.png", "_overlay_"+str(kclass)+"prob.png")

            if N_DATA_BANDS <= 3:
                plt.imshow(bigimage, cmap='gray')
            else:
                plt.imshow(bigimage[:, :, :3])

            plt.imshow(softmax_scores[:,:,kclass], alpha=0.5, vmax=1, vmin=0)
            plt.axis("off")
            plt.colorbar()
            plt.savefig(tmpfile, dpi=200, bbox_inches="tight")
            plt.close("all")



def compute_segmentation(
    TARGET_SIZE: tuple,
    N_DATA_BANDS: int,
    NCLASSES: int,
    MODEL,
    sample_direc: str,
    model_list: list,
    metadatadict: dict,
    profile: str,
    out_dir_name: str
) -> None:
    """applies models in model_list to directory of imagery in sample_direc.
    imagery will be resized to TARGET_SIZE and should contain number of bands specified by
    N_DATA_BANDS. The outputted segmentation will contain number of classes corresponding to NCLASSES.
    Outputted segmented images will be located in a new subdirectory named 'out' created within sample_direc.
    Args:
        TARGET_SIZE (tuple):imagery will be resized to this size
        N_DATA_BANDS (int): number of bands in imagery
        NCLASSES (int): number of classes used in segmentation model
        sample_direc (str): full path to directory containing imagery to segment
        model_list (list): list of loaded models
        metadatadict (dict): config files, model weight files, and names of each model in model_list
    """
    # look for TTA config
    if "TESTTIMEAUG" not in locals():
        TESTTIMEAUG = False
    WRITE_MODELMETADATA = False
    OTSU_THRESHOLD=False

    # Read in the image filenames as either .npz,.jpg, or .png
    files_to_segment = sort_files(sample_direc)
    sample_direc=os.path.abspath(sample_direc)
    # Compute the segmentation for each of the files

    for file_to_seg in tqdm.auto.tqdm(files_to_segment):
        do_seg(
            file_to_seg,
            model_list,
            metadatadict,
            MODEL,
            sample_direc=sample_direc,
            NCLASSES=NCLASSES,
            N_DATA_BANDS=N_DATA_BANDS,
            TARGET_SIZE=TARGET_SIZE,
            TESTTIMEAUG=TESTTIMEAUG,
            WRITE_MODELMETADATA=WRITE_MODELMETADATA,
            OTSU_THRESHOLD=OTSU_THRESHOLD,
            profile=profile,
            out_dir_name=out_dir_name
        )