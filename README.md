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
* Segmentation Zoo is a repository of image segmentation models for broad use in the geosciences, pre-trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym). These models may be used for general scientific enquiries or for use in downstream applications (credit is always appreciated!)
* 'Broad use' is open to interpretation, but perhaps relates to 'broad classes' of widespread utility in geosciences (including geology, ecology, hydrology, oceanography and all related fields) such as water, sediment, vegetation, soil, rock, and landcover types. Such broad classes may also be broken down into other generally useful classes, such as water types (e.g. whitewater, turbid water), sediment types (sand, gravel, etc), vegetation types, soil types, and rock types. Examples of broad, established ontologies for geoscience disciplines are [CMECS](https://iocm.noaa.gov/standards/cmecs-home.html) and [GLanCE](https://sites.bu.edu/measures/project-methods/land-cover-classification-system/)
* Segmentation Zoo promotes use of models published using Zenodo, along with a model card in a specific format that documents the model, its construction, and intendd uses.
* Finally, 'Zoo contains various examples of model implementations. 'Implementation' in this scope refers to the use of a model on unseen sample imagery. There are a number of ways that this may be acheived, and would differ depending on factors such as the type of imagery, amount of overlap in imagery, required accuracy, the need for probabilistic outputs, number of classes, etc. Whereas 'Segmentation Gym' contains a basic model implementation (called `seg_images_in_folder.py`), bespoke model applications may differ considerably. The notebooks and scripts folders in Zoo contain basic and more advanced examples of how to use a model for segmentation, including use of ensemble models. We also hope to demonstrate possible transfer learning scenarios 

## ✍️ Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)

Contributions:
* [@ebgoldstein](https://github.com/ebgoldstein)
* [@2320sharon](https://github.com/2320sharon)

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

## Segmentation Zoo User Roles

1. User of models
    - We anticipate that most Zoo users will simply want to use published models listed in the wiki, and adapt the example notebook workflows for their own purposes
2. Contributor of models
    - We welcome contributions of Gym models! The basic steps (to be outlined in the wiki) involve a) making a new Zenodo release including your model wrights file, config file, and modelcard file; b) cloning the Zoo wiki and adding a page explaining the model's purpose; and c) issuing a pull request for review by Doodleverse HQ
3. Contributor of implementation workflows
    - We welcome contributions of Gym model implementations! The basic steps (to be outlined in the wiki) involve a) cloning the Zoo repo and adding a notebook or script showing your workflow; and b) issuing a pull request for review by Doodleverse HQ

## Quick start guide

To run the example notebooks, change directory (`cd`) to the `notebooks` directory and launch jupyter using

`jupyter notebook`

The menu of notebooks can be accessed in the browser at `http://localhost:8888/tree`

You should adapt the workflows shown in these notebooks to your own imagery

## User guide

Check out our [wiki](https://github.com/Doodleverse/segmentation_zoo/wiki) for further instructions