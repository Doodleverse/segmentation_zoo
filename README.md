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


## User guide

Check out our [wiki](https://github.com/Doodleverse/segmentation_zoo/wiki) for further instructions