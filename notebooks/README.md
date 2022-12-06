## Example notebooks

This folder has some example notebooks for working with models and inspecting outputs

Contents:

1. `SingleImagePrediction.ipynb` by Evan Goldstein
    - Exemplifies loading a model and using it for prediction on a single image
    - Shows how to compute and visualize margin of prediction (confidence in model prediction, which is the distance to the decision threshold)
    - Shows how to carry out basic test-time augmentation

2. `EnsemblePrediction_2class.ipynb` by Daniel Buscombe
This notebook walks through loading several models each trained for the same task, making ensemble prediction with an image, and inspecting the outputs

* Part 1: load libraries
* Part 2: select and load models
    * we use 3 models trained for the same task: finding diftwood in aerial images of a river
    * we repurpose a snippet of code from the Gym script `seg_images_in_folder.py` to load in and make each model and apply the model weights from h5 files
* Part 3: Application of each model on a sample image
    * load an example image
    * apply all 3 models to the image and inspect the outputs
    * visualize the variability in the outputs
* Part 4: Application of ensembled model on a sample image
    * apply all 3 models and aggregate softmax scores for an ensemble model prediction
    * demonstrate the use of Otsu (adaptive) thresholding versus normal thresholding
    * demonstrate the use of Conditional Random Field for model inference (no thresholding)
* Part 5: Putting it all together: Ensembled model with TTA and  Otsu thresholding
    * create functions for reading imagery and applying models with or without Otsu thresholding
    * apply the model to a folder of 20 images to examine variability in model outputs, and to visually determine the optimal inference strategy

3. `HyperEnsemblePrediction_binary_and_multiclass_model_merging.ipynb` by Daniel Buscombe
This notebook walks through applying several models each trained for different tasks, but containing common classes. This notebook follows-on from the `EnsemblePrediction_2class.ipynb` notebook (you should read that one first)

One set of models are trained to find wood in aerial imagery of rivers (these are binary models with NCLASSES=2, i.e. wood and not wood). Another set of models are trained to find 4 classes (water, sediment, wood, other). A final set of models are trained to find 5 classes (water, sediment, wood, vegetation, other). We combine all models to make a "hyper-ensemble" prediction with an image, and inspecting the outputs

* Part 1: load libraries
* Part 2: select and load models
    * one wood model, plus one 4-class model, plus one 5-class model
    * common across all 3 models class is wood
    * common class between 4- and 5-class models are a) water, b) sediment, c) wood, and d) other
* Part 3: apply each model separately to a folder of images
    * we'll see how well each model performs separately across a folder of images
* Part 4: combine models and use to predict on a folder of images
    * we combine the 3 models in such a way that 2 ensembled outputs are created, a) wood only, and b) a 4-class
    * this is done by merging softmax scores for each class, then argmaxing each, then combining into a multiclass output

4. `SDSmodels.ipynb` by Daniel Buscombe
This notebook walks through loading several models each trained for the same task, making ensemble prediction with an image, and inspecting the outputs

* Part 1: load libraries
* Part 2: select and load models
    * we use several models trained for the same task: finding the coastal shoreline
    * we repurpose a snippet of code from the Gym script `seg_images_in_folder.py` to load in and make each model and apply the model weights from h5 files
* Part 3: Application of each model for 2-classes
    * apply all models and aggregate softmax scores for an ensemble model prediction
    * Otsu (adaptive) thresholding versus normal thresholding
    * TTA versus no TTA (test-time augmentation)
* Part 4: Application of each model for 4-classes remapped to 2-classes
    * apply all models and aggregate softmax scores for an ensemble model prediction
    * Otsu (adaptive) thresholding versus normal thresholding
    * TTA versus no TTA (test-time augmentation)