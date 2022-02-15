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
* [@Doodleverse](https://github.com/Doodleverse) Marda Science / USGS Pacific Coastal and Marine Science Center.

Contributions:
* [@2320sharon](https://github.com/2320sharon)
* [@ebgoldstein](https://github.com/ebgoldstein)

We welcome collaboration! Please use our [Discussions tab](https://github.com/Doodleverse/segmentation_zoo/discussions) if you're interested in this project. We welcome user-contributed models trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym), but please wait for announcements while we get our act together, and meanwhile we'll be happy to chat about your data, models, science, art, and life in general in [Discussions](https://github.com/Doodleverse/segmentation_zoo/discussions).

## :eight_spoked_asterisk: Models

### :floppy_disk: Menu
1. 'landsat': optical (RGB), near-infrared (NIR) and short-wave-infrared (SWIR) satellite imagery and 5 classes
2. 'aerial': optical (RGB) oblique photographs of sandy and rocky shoreline environments and two classes (water, land)
3. ...more are coming. Kindly consider contributing yours.

'landsat' are the models that made the forthcoming paper: Buscombe and Goldstein (in prep.) "A Reproducible and Reusable Pipeline for Segmentation of Geoscientific Imagery." intended for Earth and Space Science. The models were trained on a subset of the "Coast Train" dataset, described in the forthcoming manuscript Buscombe et al (in prep) "A 1.2 Billion Pixel Human-Labeled Dataset for Data-Driven Classification of Coastal Environments" intended for Scientific Data.

### :orange_book: Folder organization 
Folders are organized with the following structure:

--> {imagery type}
----> {model type}
------> {data bands}

The model is the `config` and `weights` folders collectively, which should be mirrored in organization like above. For example, there are 'landsat' models based on both UNets and Residual UNets, and for 3-band (RGB) and 5-band (RGB+NIR+SWIR) inputs, so the models are organized as follows:

--> landsat
----> resunet
------> RGB
------> RGB-NIR-SWIR
----> unet
------> RGB
------> RGB-NIR-SWIR

For each model, there are 3 files: 
1. config file: this is the file that was used by [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) to create the weights file. It contains instructions for how to make the model and the data it used, as well as instructions for how to use the model for prediction. It is a handy wee thing and mastering it means mastering the entire Doddleverse. It is needed by the [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) function [`seg_images_in_folder.py`](https://github.com/Doodleverse/segmentation_gym/blob/main/seg_images_in_folder.py) to segment a folder of images, along with the weights file described below

2. weights file: this is the file that was created by the [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) function [`train_model.py`](https://github.com/Doodleverse/segmentation_gym/blob/main/train_model.py). It contains the trained model's parameter weights. It can called by the [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) function [`seg_images_in_folder.py`](https://github.com/Doodleverse/segmentation_gym/blob/main/seg_images_in_folder.py) to segment a folder of images

3. classes file: this is a text file containing the names of the classes




