#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:25:58 2019

@author: jwitmer
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Activation
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import backend as K

import numpy as np
from dotenv import load_dotenv
import matplotlib as plt
import h5py
import os
from sklearn import metrics
from pathlib import Path
from random import shuffle
import pickle as pkl


load_dotenv()

AUDIOSET_SPLITS_V1 = os.getenv('AUDIOSET_SPLITS_V1')
AUDIOSET_SPLITS_V1 = Path(AUDIOSET_SPLITS_V1).expanduser()



#%%
filtered_dev_h5 = h5py.File(str(AUDIOSET_SPLITS_V1 / 'filter2.h5'), 'r')
X_dev = filtered_dev_h5['X'][()]
Y_dev = filtered_dev_h5['y'][()].astype(int)

X_dev = X_dev/255
X_dev = np.reshape(X_dev, (X_dev.shape[0], X_dev.shape[1], X_dev.shape[2], 1))


classes_to_keep = [ 1, 2, 111, 288, 343, 396]
N_classes = len(classes_to_keep)

# only keep labels for classes we care about
Y_dev = Y_dev[:,classes_to_keep]
n_classes_per_example = np.sum(Y_dev,axis=1,keepdims=True)

print("Total number of examples in dev set:")
print(Y_dev.shape[0])
n_examples_per_class= np.sum(Y_dev, axis = 0, keepdims = True)
print("Number of examples per class in dev set: ")
print(n_examples_per_class)



#%%
# 
model = keras.models.load_model('audioset_multilabel_M15.h5')

predictions = model.predict(X_dev)
predictions_thresholded = predictions > 0.5

f1_scores = metrics.f1_score(Y_dev, predictions_thresholded, average=None)
precisions = metrics.precision_score(Y_dev, predictions_thresholded, average=None)
recalls = metrics.recall_score(Y_dev, predictions_thresholded, average=None)
accuracy = metrics.accuracy_score(Y_dev, predictions_thresholded)


print('=========================================================')
print('PERFORMANCE METRICS')
print('---------------------------------------------------------')

print('F1 scores:')
print(f1_scores)
print('Average: ' + str(np.mean(f1_scores)))

print('Precision:')
print(precisions)  
print('Average: ' + str(np.mean(precisions)))

print('Recalls:')
print(recalls)
print('Average: ' + str(np.mean(recalls)))

print('Accuracy:')
print(accuracy)

print('=========================================================')






