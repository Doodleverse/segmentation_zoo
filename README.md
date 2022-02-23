# :package: Segmentation Zoo :elephant:
[![Last Commit](https://img.shields.io/github/last-commit/Doodleverse/segmentation_zoo)](
https://github.com/Doodleverse/segmentation_zoo/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Doodleverse/segmentation_zoo/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/Doodleverse/segmentation_zoo/wiki)
![GitHub](https://img.shields.io/github/license/Doodleverse/segmentation_zoo)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/Doodleverse/segmentation_zoo/discussions)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)


![zoo](https://user-images.githubusercontent.com/3596509/153691807-1da4d3ba-377b-40af-9891-c469cc6390c1.png)

Hi! This is work in progress - please check back later or use our [Discussions tab](https://github.com/Doodleverse/segmentation_zoo/discussions) if you're interested in this project and would like to say hi. The models here should be considered beta, and could improve over time, or they may change in structure, so any usage of these models should be for practice with [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym). Better documentation is forthcoming.

## :star2: Highlights
* Segmentation Zoo is a repository of image segmentation models, pre-trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym)
* Contains models that made the forthcoming paper: Buscombe and Goldstein (in prep.) "A Reproducible and Reusable Pipeline for Segmentation of Geoscientific Imagery." intended for Earth and Space Science. The models were trained on a subset of the "Coast Train" dataset, described in the forthcoming manuscript Buscombe et al (in prep) "A 1.2 Billion Pixel Human-Labeled Dataset for Data-Driven Classification of Coastal Environments" intended for Scientific Data.
* It serves as a convenient place to access models for tasks involving simple classification schema on common data types such as Landsat-8 scenes and oblique photographs. We are not promoting these models for universal application for these classification tasks. These models may have some practical usage but are provided primarily for illustrative purposes.
* Usage of these models is encouraged for practice with [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) to see the organization and usage of models created by that toolset
* We hope to eventually provide documentation exemplifying the potential uses of models - please check back later or watch this repository

## ✍️ Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs) Marda Science / USGS Pacific Coastal and Marine Science Center.

Contributions:
* [@2320sharon](https://github.com/2320sharon)
* [@ebgoldstein](https://github.com/ebgoldstein)

We welcome collaboration! Please use our [Discussions tab](https://github.com/Doodleverse/segmentation_zoo/discussions) if you're interested in this project. We welcome user-contributed models trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym), but please wait for announcements while we get our act together, and meanwhile we'll be happy to chat about your data, models, science, art, and life in general in [Discussions](https://github.com/Doodleverse/segmentation_zoo/discussions).


## ⬇️ Installation

If you already have the conda environment, `gym`, installed from [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym), you may use that. Otherwise, we advise creating a new conda environment to run the program.

1. Clone the repo:

```
git clone --depth 1 https://github.com/Doodleverse/segmentation_zoo.git
```

(`--depth 1` means "give me only the present code, not the whole history of git commits" - this saves disk space, and time)

2. Create a conda environment called `zoo`

```
conda env create --file install/zoo.yml
conda activate zoo
```

If you get errors associated with loading the model weights you may need to:

```
pip install "h5py==2.10.0" --force-reinstall
```

and just ignore any errors.


## :eight_spoked_asterisk: Models

### :floppy_disk: Menu

Segmentation Zoo allows you to apply models that have been pre-trained on labeled imagery. Those models are hosted on Zenodo.

#### Landsat models
Imagery must be in 8-bit jpeg format. These are the models that made the forthcoming paper: Buscombe and Goldstein (in prep.) "A Reproducible and Reusable Pipeline for Segmentation of Geoscientific Imagery." intended for Earth and Space Science. The models were trained on a subset of the "Coast Train" dataset, described in the forthcoming manuscript Buscombe et al (in prep) "A 1.2 Billion Pixel Human-Labeled Dataset for Data-Driven Classification of Coastal Environments" intended for Scientific Data.

`landsat_6229071` contains Residual UNets whereas `landsat_6230083` contains UNets 

##### landsat_6229071
Zenodo release: https://zenodo.org/record/6229071#.YhVhafuIZhE

For Landsat-8 and similar satellite scenes. Res-UNet model sets available for the following imagery types:
1. 3-band: optical (RGB)
2. 5-band: combined optical (RGB) near-infrared (NIR) and short-wave-infrared (SWIR)
3. 1-band: modified normalized water index (MNDWI)
4. 1-band: normalized water index (NDWI)

and the following classes:
0. null
1. water (unbroken water)
2. whitewater (surf, active wave breaking)
3. sediment (natural deposits of sand. gravel, mud, etc)
4. other (development, bare terrain, vegetated terrain, etc)


##### landsat_6230083
Zenodo release: https://zenodo.org/record/6230083#.YhVptvuIZhE

For Landsat-8 and similar satellite scenes. UNet model sets available for the following imagery types:
1. 3-band: optical (RGB)
2. 5-band: combined optical (RGB) near-infrared (NIR) and short-wave-infrared (SWIR)
3. 1-band: modified normalized water index (MNDWI)
4. 1-band: normalized water index (NDWI)

and the following classes:
0. null
1. water (unbroken water)
2. whitewater (surf, active wave breaking)
3. sediment (natural deposits of sand. gravel, mud, etc)
4. other (development, bare terrain, vegetated terrain, etc)


#### Sand/coin image models
Imagery must be in 8-bit jpeg format. 

##### coin_6229579
Zenodo release: https://zenodo.org/record/6229579#.YhWk0_uIZhE

For images of sand with a coin for scale. For finding the coin. Res-UNet model sets available for the following imagery types:
1. 3-band: optical (RGB)

and the following classes:
0. other
1. coin


#### Aerial shoreline image models
Imagery must be in 8-bit jpeg format. `aerial_6234122` contains Residual UNets whereas `aerial_6235090` contains UNets 

##### aerial_6234122
Zenodo release: https://zenodo.org/record/6234122#.YhWiPfuIZhE

For oblique aerial images of coasts. For finding the water. Res-UNet model sets available for the following imagery types:
1. 3-band: optical (RGB)

and the following classes:
0. water
1. land

##### aerial_6235090
Zenodo release: https://zenodo.org/record/6235090#.YhWjH_uIZhE

For oblique aerial images of coasts. For finding the water. UNet model sets available for the following imagery types:
1. 3-band: optical (RGB)

and the following classes:
0. water
1. land


## :orange_book: Folder organization 

For each model, there are 3 files: 
1. config file: this is the file that was used by [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) to create the weights file. It contains instructions for how to make the model and the data it used, as well as instructions for how to use the model for prediction. It is a handy wee thing and mastering it means mastering the entire Doodleverse. It is needed by the [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) function [`seg_images_in_folder.py`](https://github.com/Doodleverse/segmentation_gym/blob/main/seg_images_in_folder.py) to segment a folder of images, along with the weights file described below

2. weights file: this is the file that was created by the [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) function [`train_model.py`](https://github.com/Doodleverse/segmentation_gym/blob/main/train_model.py). It contains the trained model's parameter weights. It can called by the [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) function [`seg_images_in_folder.py`](https://github.com/Doodleverse/segmentation_gym/blob/main/seg_images_in_folder.py) or the [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) function [`select_model_and_batch_process_folder.py`](https://github.com/Doodleverse/segmentation_zoo/scripts/blob/main/select_model_and_batch_process_folder.py) to segment a folder of images

3. model card file: this is a json file containing the following fields that collectively describe the model origins, training choices, and dataset that the model is based upon. There is some redundancy between this file and the `config` file (described above) that contains the instructions for the model training and implementation. The model card file is not used by the program but is important metadata

```
{
	"DETAILS": {
	"NAME": "name of model set",
    "DATE": "YYYY-MM-DD",
    "URL":"string",
    "CITATION":"Bloggs, J. (2022) Name. Zenodo data release XXXXXXX",
    "QUERIES": "email or website",
    "CREDIT":"Name",
    "INTENDED_PURPOSE":"Description",
    "KEYWORDS": {
      "1": "keyword 1",
      "2": "keyword 2"
    }
	},  
	"DATASET": {
	"NAME": "Dataset name", 
    "SOURCE": "dataset url or description of how to obtain data",
    "CITATION": "Bloggs et al 2022, Name of dataset: Publisher, URL",
    "NUMBER_LABELED_IMAGES": 999999,
		"CLASSES": {
			"0": "class 1",
			"1": "class 2",
			"2": "class 3"
		},
		"ORIG_CLASSES": {
			"0": "orig class 1",
			"1": "orig class 2",
			"2": "orig class 3",
			"3": "orig class 4"
		},
		"REMAP_CLASSES": {
			"0": 0,
			"1": 2,
			"2": 4,
			"3": 6
		},
		"N_DATA_BANDS": 3,
    "BAND_NAMES": {
      "0": "red",
      "1": "green",
      "2": "blue"
    }
	},
	"MODEL": {
	"NAME": "resunet",
    "KERNEL":7,
    "STRIDE":2,
    "FILTERS":6    
	},
	"TRAINING": {
    "BATCH_SIZE": 8,
    "DROPOUT":0.1,
    "DROPOUT_CHANGE_PER_LAYER":0.0,
    "DROPOUT_TYPE":"standard",
    "USE_DROPOUT_ON_UPSAMPLING":false,
    "LOSS":"cat",
    "PATIENCE": 10,
    "MAX_EPOCHS": 100,
    "VALIDATION_SPLIT": 0.6,
    "RAMPUP_EPOCHS": 20,
    "SUSTAIN_EPOCHS": 0.0,
    "EXP_DECAY": 0.9,
    "START_LR":  1e-7,
    "MIN_LR": 1e-7,
    "MAX_LR": 1e-4   
	},
	"AUGMENTATION": {
    "AUGMENTATION_USED": true,
    "FILTER_VALUE": 0,
    "AUG_ROT": 0.05,
    "AUG_ZOOM": 0.05,
    "AUG_WIDTHSHIFT": 0.05,
    "AUG_HEIGHTSHIFT": 0.05,
    "AUG_HFLIP": true,
    "AUG_VFLIP": false,
    "AUG_LOOPS": 10,
    "AUG_COPIES": 5  
	}     
}
```


## :rainbow: What to do 


### Example imagery
If you need some sample imagery to work with, from the root directory,

```
cd utilities
python download_sample_data.py
```

It will create the folder `sample_data` if it doesn't already exist, then download the sample data of the dataset you choose from the drop-down menu

To create 5-band inputs, use ```utilities/merge_nd_inputs4pred.py```. To create MNDWI imagery, use ```utilities/make_mndwi_4pred.py``` or to create NDWI imagery, use ```utilities/make_ndwi_4pred.py```. Follow the prompts asking for various inputs, and the program will create .npz format files containing imagery. You may subsequently use these .npz format files for prediction. Note that this is not necessary for 3-band jpeg images. 


### Segment imagery (apply model to sample imagery)
From the root directory,

```
cd scripts
python select_model_and_batch_process_folder.py
```

It will create the folder `downloaded_models` if it doesn't already exist, then download the model of the dataset you choose from the first drop-down menu if that doesn't already exist. Then you are prompted to choose a dataset, which relates to the type of imagery the model was trained using and what the model should be applied to. If you choose the dataset 'ALL', the program will download models for all imagery types and then exit. Otherwise, select your imagery type from the menu.

In the next menu, select 'BEST' to apply only the best model, or 'ENSEMBLE' to create an ensemble prediction. We usually would recommend an ensemble model, however you should determine which is best for your task.

The final menu asks you to select a directory of sample images (or npzs) to segment. Model outputs are written to a subdirectory of the sample imagery called 'out'.

