# Written by Dr Daniel Buscombe, Marda Science LLC
# for  the USGS Coastal Change Hazards Program
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
# OUT OF OR IN CONNECTION WITH THE zSOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# functions adapted minimally from https://github.com/zhiminwang1/Remote-Sensing-Image-Segmentation/blob/master/seg_metrics.py
# contribution credits added to the README and Doodleverse/segmentation_gym README

import numpy as np
import tensorflow as tf

#=================================================
def Precision(confusionMatrix):  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision  

#=================================================
def Recall(confusionMatrix):
    epsilon = 1e-6
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis = 0) + epsilon)
    return recall

#=================================================
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score

#=================================================
def IntersectionOverUnion(confusionMatrix):  
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

#=================================================
def MeanIntersectionOverUnion(confusionMatrix):  
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU

#=================================================
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

#=================================================
def ConfusionMatrix(numClass, imgPredict, Label): 
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix
    
#=================================================
def OverallAccuracy(confusionMatrix):  
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA

#=================================================
def MatthewsCorrelationCoefficient(confusionMatrix):  

    t_sum = tf.reduce_sum(confusionMatrix, axis=1)
    p_sum = tf.reduce_sum(confusionMatrix, axis=0)

    n_correct = tf.linalg.trace(confusionMatrix)
    n_samples = tf.reduce_sum(p_sum)

    cov_ytyp = n_correct * n_samples - tf.tensordot(t_sum, p_sum, axes=1)
    cov_ypyp = n_samples ** 2 - tf.tensordot(p_sum, p_sum, axes=1)
    cov_ytyt = n_samples ** 2 - tf.tensordot(t_sum, t_sum, axes=1)

    cov_ytyp = tf.cast(cov_ytyp,'float')
    cov_ytyt = tf.cast(cov_ytyt,'float')
    cov_ypyp = tf.cast(cov_ypyp,'float')

    mcc = cov_ytyp / tf.math.sqrt(cov_ytyt * cov_ypyp)
    if tf.math.is_nan(mcc ) :
        mcc = tf.constant(0, dtype='float')
    return mcc.numpy()

#=================================================
def AllMetrics(numClass, imgPredict, Label): 

    confusionMatrix = ConfusionMatrix(numClass, imgPredict, Label)
    OA = OverallAccuracy(confusionMatrix)
    FWIoU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIoU = MeanIntersectionOverUnion(confusionMatrix)
    f1score = F1Score(confusionMatrix)
    recall = Recall(confusionMatrix)
    precision = Precision(confusionMatrix)
    mcc = MatthewsCorrelationCoefficient(confusionMatrix)

    return {"OverallAccuracy":OA, 
            "Frequency_Weighted_Intersection_over_Union":FWIoU, 
            "MeanIntersectionOverUnion":mIoU, 
            "F1Score":f1score, 
            "Recall":recall, 
            "Precision":precision,
            "MatthewsCorrelationCoefficient": mcc}