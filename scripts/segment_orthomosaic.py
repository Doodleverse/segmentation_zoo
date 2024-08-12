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
from tkinter import filedialog, messagebox
from tkinter import *
import sys, os, shutil, json
from glob import glob
## geospatial imports
from osgeo import gdal
gdal.SetCacheMax(2**30) # max out the cache

# local imports
import model_data_funcs
import model_inference_funcs
from model_inference_funcs import tf

###### user variables
# do_parallel = True ## use GPU
do_parallel = False ## dont use GPU

# profile = 'full' ## predseg + meta +overlay
profile = 'meta' ## predseg + meta 
# profile = 'minimal' ## predseg
## profile must be 'meta' or 'full' for this script to work

make_RGB_label_ortho = True # make an RGB label mosaic as well as a greyscale one
make_jpeg = True ## make JPEG mosaics as well as geotiffs
make_probs = True ##make probability rasters per class

#===============================
if do_parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":

    ##################################
    ##### STEP 1: OPTIONS

    ### choose target size 
    root = Tk()
    root.geometry('200x100')

    choices = [
        "256",
        "512",
        "768",
        "1024",
        "1536",
        "2048"
    ]

    variable = StringVar(root)
    variable.set("768")
    w = OptionMenu(root, variable, *choices)
    w.pack()
    root.mainloop()

    TARGET_SIZE = int(variable.get())
    print("You chose TARGET_SIZE : {}".format(TARGET_SIZE))


    ### choose resampleAlg 
    root = Tk()
    root.geometry('200x100')

    choices = [
        "nearest",
        "mode",
        "min",
        "max",
        "average",
        "gauss"
    ]

    variable = StringVar(root)
    variable.set("mode")
    w = OptionMenu(root, variable, *choices)
    w.pack()
    root.mainloop()

    resampleAlg = variable.get()
    print("You chose resample algorithm : {}".format(resampleAlg))

    #### choose generic task 
    root = Tk()
    root.geometry('200x100')

    ## only these two apply to orthomosaics
    choices = [
        "generic_landcover_highres",
        "coastal_landcover_highres",
        "custom"
    ]

    variable = StringVar(root)
    variable.set("generic_landcover_highres")
    w = OptionMenu(root, variable, *choices)
    w.pack()
    root.mainloop()

    task_id = variable.get()
    print("You chose task : {}".format(task_id))

    ##### ask for a output file root name
    #### choose zenodo release

    if task_id=="generic_landcover_highres":

        root = Tk()
        root.geometry('200x100')

        choices = [
        "openearthmap_9class_7576894",
        "deepglobe_7class_7576898",
        "enviroatlas_6class_7576909",
        "aaai_building_7607895",
        "aaai_floodedbuildings_7622733",
        "xbd_building_7613212",
        "xbd_damagedbuilding_7613175",
        "elwha_alluvial_driftwood_segformer_7933013",
        "elwha_alluvial_sediment_segformer_8172908",
        "elwha_alluvial_4class_segformer_8172858"
        ]

        variable = StringVar(root)
        variable.set("openearthmap_9class_7576894")

    elif task_id=="coastal_landcover_highres":

        root = Tk()
        root.geometry('200x100')

        choices = [
            "orthoCT_2class_7574784", 
            "orthoCT_5class_7566992",
            "orthoCT_5class_segformer_7641708",
            "orthoCT_8class_7570583",
            "orthoCT_8class_segformer_7641724",
            "chesapeake_7class_7576904",
            "chesapeake_7class_segformer_7677506"
            "watermask_noaa_oblique_2class_7604083",
            "watermask_aerial_oblique_2class_7604075",
            "watermask_aerial_nadir_2class_7604077",
            "watermask_pcmsc_wm_v2_7700430"            
        ]

        variable = StringVar(root)
        variable.set("orthoCT_5class_7566992")

    elif task_id=="custom":

        root = Tk()
        root.filename =  filedialog.askopenfilename(title = "Select first weights file",filetypes = (("h5 file","*.h5"),("all files","*.*")))
        weights = root.filename
        print(weights)
        root.withdraw()

        # weights_files : list containing all the weight files fill paths
        weights_files=[]
        weights_files.append(weights)

        from tkinter import messagebox

        # Prompt user for more model weights and appends them to the list W that contains all the weights
        result = 'yes'
        while result == 'yes':
            result = messagebox.askquestion("More Weights files?", "More Weights files?", icon='warning')
            if result == 'yes':
                root = Tk()
                root.filename =  filedialog.askopenfilename(title = "Select weights file",filetypes = (("weights file","*.h5"),("all files","*.*")))
                weights = root.filename
                root.withdraw()
                weights_files.append(weights)

    #=============================

    if task_id!="custom":

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

        # ####======================================

        # segmentation zoo directory
        parent_direc = os.path.dirname(os.getcwd())
        # create downloaded models directory in segmentation_zoo/downloaded_models
        downloaded_models_dir = get_models_dir = model_data_funcs.get_model_dir(parent_direc, "downloaded_models")
        print(f"Downloaded Models Located at: {downloaded_models_dir}")
        # directory to hold specific downloaded model
        model_direc = model_data_funcs.get_model_dir(downloaded_models_dir, dataset_id)

        # get list of available files to download for zenodo id
        files = model_data_funcs.request_available_files(zenodo_id)
        # print(f"Available files for zenodo {zenodo_id}: {files}")

        zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
        # check if zenodo release contains zip file 'rgb.zip'
        is_zip = model_data_funcs.is_zipped_release(files)
        # zenodo release contained file 'rgb.zip' download it and unzip it
        if is_zip:
            print("Checking for zipped model")
            zip_url = zipped_model_list[0]["links"]["self"]
            model_direc = model_data_funcs.download_zipped_model(model_direc, zip_url)
        # zenodo release contained no zip files. perform async download
        elif is_zip == False:
            if model_choice == "BEST":
                model_data_funcs.download_BEST_model(files, model_direc)
            elif model_choice == "ENSEMBLE":
                model_data_funcs.download_ENSEMBLE_model(files, model_direc)

    else:
        zenodo_id = 'custom'

        # ###==============================================

    ###############################################
    ################# INPUTS
    ### user inputs
    # Request the orthomosaic geotiff file
    root = Tk()
    root.filename =  filedialog.askopenfilename(title = "Select orthomosaic file",filetypes = (("geotff file","*.tif"),("jpeg file (with xml and/or wld)","*.jpg"),("all files","*.*")))
    image_ortho = root.filename
    print(image_ortho)
    root.withdraw()

    OVERLAP_PX = TARGET_SIZE//2
    print("Overlap size : {} px".format(OVERLAP_PX))

    #=================================================
    O = []
    O.append(image_ortho)

    result = 'yes'
    while result == 'yes':
        result = messagebox.askquestion("More jobs?", "More jobs?", icon='warning')
        if result == 'yes':

            ###############################################
            ################# INPUTS
            ### user inputs
            # Request the orthomosaic geotiff file
            root = Tk()
            root.filename =  filedialog.askopenfilename(title = "Select orthomosaic file",filetypes = (("geotff file","*.tif"),("jpeg file (with xml and/or wld)","*.jpg"),("all files","*.*")))
            image_ortho = root.filename
            print(image_ortho)
            root.withdraw()

            O.append(image_ortho)

    print(O)

    ##################################
    ##### STEP 2: MAKE ORTHO TILES

    ###############################################
    ################# ORTHO TILES
    ### make ortho tiles with overlap from the mosaic image
    for image_ortho in O:
        print(image_ortho)
        indir = os.path.dirname(image_ortho)
        outdir = indir+os.sep+'tiles'

        if os.path.isdir(os.path.normpath(outdir)):
            shutil.rmtree(outdir)
            os.mkdir(outdir)

        else:
            os.mkdir(outdir)

        ### chop up image ortho into tiles with 50% overlap

        if os.name == "nt":

            try:
                print("Attempting to use gdal from conda env")
                cmd = 'python gdal_retile.py -r near -ot Byte -ps {} {} -overlap {} -co "tiled=YES" -targetDir {} {}'.format(TARGET_SIZE,TARGET_SIZE,OVERLAP_PX,outdir,image_ortho)
                os.system(cmd)
            except:
                print("That didn't work. Attempting to use gdal from osgeo4w")
                from subprocess import Popen, PIPE
                process=Popen(["python","C:\\OSGeo4W64\\bin\\gdal_retile.py","-r", "near", "-ot", "Byte","-ps",str(TARGET_SIZE),str(TARGET_SIZE),"-overlap",str(OVERLAP_PX),"-co", "tiled=YES","-targetDir",outdir, image_ortho], stdout=PIPE, stderr=PIPE)
                stdout, stderr = process.communicate()

        else:
            try:
                ## it would be cleaner if the gdal_retile.py script could be wrapped in gdal/osgeo python, but it errored for me ...
                cmd = 'gdal_retile.py -r near -ot Byte -ps {} {} -overlap {} -co "tiled=YES" -targetDir {} {}'.format(TARGET_SIZE,TARGET_SIZE,OVERLAP_PX,outdir,image_ortho)
                os.system(cmd)
            except:
                cmd = 'python gdal_retile.py -r near -ot Byte -ps {} {} -overlap {} -co "tiled=YES" -targetDir {} {}'.format(TARGET_SIZE,TARGET_SIZE,OVERLAP_PX,outdir,image_ortho)
                os.system(cmd)

        ### convert to jpegs for Zoo model
        kwargs = {
            'format': 'JPEG',
            'outputType': gdal.GDT_Byte
        }

        def gdal_translate_jpeg(f, bandList, kwargs):
            ds = gdal.Translate(f.replace('.tif','.jpg'), f, bandList=bandList, **kwargs)
            ds = None # close and save ds

        files_to_convert = glob(outdir+os.sep+'*.tif')

        bandList=[1,2,3]

        if len(files_to_convert)>0:

            for f in files_to_convert:
                gdal_translate_jpeg(f, bandList, kwargs)

            ## delete tif files
            _ = [os.remove(k) for k in glob(outdir+os.sep+'*.tif')]

        else:
            print("No tif files found")
            sys.exit(0)

        ##################################
        ##### STEP 3: MAKE LABEL TILES

        ### apply Zoo model
        sample_direc = outdir

        if task_id!="custom":

            # weights_files : list containing all the weight files fill paths
            weights_files = model_inference_funcs.get_weights_list(model_choice, model_direc)

        # For each set of weights in weights_files load them in
        M = []
        C = []
        T = []
        for counter, weights in enumerate(weights_files):

            try:
                # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
                # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
                configfile = weights.replace("_fullmodel.h5", ".json")#.replace("weights", "config").strip()
                with open(configfile) as file:
                    config = json.load(file)
            except:
                # Turn the .h5 file into a json so that the data can be loaded into dynamic variables
                configfile = weights.replace(".h5", ".json")#.replace("weights", "config").strip()
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
                model, model_list, config_files, model_names = model_inference_funcs.get_model(weights_files)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print("Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'")
                sys.exit(2)

            # get dictionary containing all files needed to run models on data
            metadatadict = model_inference_funcs.get_metadatadict(weights_files, config_files, model_names)

        # read contents of config file into dictionary
        config = model_inference_funcs.get_config(weights_files)
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
        sample_filenames = model_inference_funcs.sort_files(sample_direc)
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

        out_dir_name = str(zenodo_id)

        # run models on imagery
        try:
            print(f"file: {file}")
            model_inference_funcs.compute_segmentation(
                TARGET_SIZE,
                N_DATA_BANDS,
                NCLASSES,
                MODEL,
                sample_direc,
                model_list,
                metadatadict,
                profile,
                out_dir_name
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(f"{file} failed. Check config file, and check the path provided contains valid imagery")


        if len(TARGET_SIZE)>1:
            TARGET_SIZE = TARGET_SIZE[0]
        print(TARGET_SIZE)

        ##################################
        ##### STEP 4: STITCH ORTHO LABEL TILES
        ### now, you have the 'out' folder ...

        ## the out folder may contain a bunch of crap we dont want. let's delete those files
        ## this should do nothing if mode='meta' or mode='minimal'

        # "prob.png" files ...
        _ = [os.remove(k) for k in glob(outdir+os.sep+out_dir_name+os.sep+'*prob.png')]

        # "overlay.png" files ...
        _ = [os.remove(k) for k in glob(outdir+os.sep+out_dir_name+os.sep+'*overlay.png')]

        ###############################################
        ################# LABEL ORTHO CREATION PREP

        ### the trick here is to make asure all the png files and xml files are
        ### in the same directoy and have the same filename root

        # Get imgs list
        imgsToMosaic = sorted(glob(os.path.join(outdir, out_dir_name, '*.png')))

        ## copy the xml files into the 'out' folder
        xml_files = sorted(glob(os.path.join(outdir, '*.xml')))

        for k in xml_files:
            shutil.copyfile(k,k.replace(outdir,outdir+os.sep+out_dir_name)) #'out'))

        ## rename pngs
        for k in imgsToMosaic:
            os.rename(k,k.replace('_predseg',''))

        xml_files = sorted(glob(os.path.join(outdir,out_dir_name, '*.xml')))
        ## rename xmls
        for k in xml_files:
            os.rename(k, k.replace('.jpg.aux.xml', '.png.aux.xml'))

        ###############################################
        ################# LABEL ORTHO CREATION 
        ### let's stitch the label "predseg" pngs!

        # make some output paths
        if make_RGB_label_ortho:
            outVRTrgb = os.path.join(indir, 'MosaicRGB.vrt')
            outTIFrgb = os.path.join(indir, 'MosaicRGB.tif')
            if make_jpeg:
                outJPGrgb = os.path.join(indir, 'MosaicRGB.jpg')

        outVRT = os.path.join(indir, 'Mosaic.vrt')
        outTIF = os.path.join(indir, 'Mosaic.tif')
        if make_jpeg:
            outJPG = os.path.join(indir, 'Mosaic.jpg')

        if make_RGB_label_ortho:
            ## now we have pngs and png.xml files with the same names in the same folder
            imgsToMosaic = sorted(glob(os.path.join(outdir, out_dir_name, '*.png')))
            print('{} images to mosaic'.format(len(imgsToMosaic)))

            # First build vrt for geotiff output
            vrt_options = gdal.BuildVRTOptions(resampleAlg=resampleAlg, srcNodata=0, VRTNodata=0)
            ds = gdal.BuildVRT(outVRTrgb, imgsToMosaic, options=vrt_options)
            ds.FlushCache()
            ds = None

            # then build tiff
            ds = gdal.Translate(destName=outTIFrgb, creationOptions=["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "TILED=YES"], srcDS=outVRTrgb)
            ds.FlushCache()
            ds = None

            if make_jpeg:
                # now build jpeg (optional)
                ds = gdal.Translate(destName=outJPGrgb, creationOptions=["QUALITY=100"], srcDS=outVRTrgb)
                ds.FlushCache()
                ds = None

        ##################################
        ##### STEP 5: MAKE AND STITCH ORTHO GREYSCALE LABEL TILES
        ## okay, now let's make the greyscale label mosaic

        npzs = sorted(glob(os.path.join(outdir, out_dir_name, '*.npz')))

        def write_greylabel_to_png(k):
            with np.load(k) as data:
                dat = 1+np.round(data['grey_label'].astype('uint8'))
            imsave(k.replace('.npz','.png'), dat, check_contrast=False, compression=0)

        for k in npzs:
            write_greylabel_to_png(k)
            
        ## now we have pngs and png.xml files with the same names in the same folder
        imgsToMosaic = sorted(glob(os.path.join(outdir, out_dir_name, '*res.png')))
        print('{} images to mosaic'.format(len(imgsToMosaic)))

        xml_files = sorted(glob(os.path.join(outdir, out_dir_name, '*.xml')))
        ## copy and name xmls
        for k in xml_files:
            shutil.copyfile(k,k.replace('.png','_res.png'))

        # First build vrt for geotiff output
        vrt_options = gdal.BuildVRTOptions(resampleAlg=resampleAlg, srcNodata=0, VRTNodata=0)
        ds = gdal.BuildVRT(outVRT, imgsToMosaic, options=vrt_options)
        ds.FlushCache()
        ds = None

        # then build tiff
        ds = gdal.Translate(destName=outTIF, creationOptions=["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "TILED=YES"], srcDS=outVRT)
        ds.FlushCache()
        ds = None

        if make_jpeg:
            # now build jpeg (optional)
            ds = gdal.Translate(destName=outJPG, creationOptions=["QUALITY=100"], srcDS=outVRT)
            ds.FlushCache()
            ds = None

        if make_probs:

            ##################################
            ##### STEP 6: MAKE AND STITCH ORTHO GREYSCALE probability TILES

            def write_greyprobs_to_tif(k):
                with np.load(k) as data:
                    dat = tf.nn.softmax(data['av_softmax_scores']).numpy().astype('float32')
                if np.ndim(dat)==2:
                    NCLASSES=1
                else:
                    NCLASSES = dat.shape[-1]
                for i in range(NCLASSES):
                    if NCLASSES>1:
                        imsave(k.replace('res.npz','prob'+str(i)+'.tif'), dat[:,:,i], check_contrast=False, compression=0)
                    else:
                        imsave(k.replace('res.npz','prob'+str(i)+'.tif'), dat, check_contrast=False, compression=0)

                return dat

            for k in npzs:
                dat = write_greyprobs_to_tif(k)

            xml_files = sorted(glob(os.path.join(outdir,out_dir_name, '*res*.xml')))
            ## copy and name xmls
            for i in range(dat.shape[-1]):
                for k in xml_files:
                    shutil.copyfile(k,k.replace('_res.png','_prob'+str(i)+'.tif'))

            if np.ndim(dat)==2:
                NCLASSES=1
            else:
                NCLASSES = dat.shape[-1]
                
            for i in range(NCLASSES):
            # for i in range(dat.shape[-1]):
                outVRT = os.path.join(indir, 'Mosaic_Prob'+str(i)+'.vrt')
                outTIF = os.path.join(indir, 'Mosaic_Prob'+str(i)+'.tif')

                ## now we have pngs and png.xml files with the same names in the same folder
                imgsToMosaic = sorted(glob(os.path.join(outdir, out_dir_name, '*prob'+str(i)+'.tif')))
                print('{} images to mosaic'.format(len(imgsToMosaic)))


                # First build vrt for geotiff output
                vrt_options = gdal.BuildVRTOptions(resampleAlg="lanczos",srcNodata=0, VRTNodata=0)
                ds = gdal.BuildVRT(outVRT, imgsToMosaic, options=vrt_options)
                ds.FlushCache()
                ds = None

                # then build tiff
                ds = gdal.Translate(destName=outTIF, creationOptions=["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "TILED=YES"], srcDS=outVRT)
                ds.FlushCache()
                ds = None

