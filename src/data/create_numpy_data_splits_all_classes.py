#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:41:17 2019

@author: jwitmer
"""

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
A simple "get-up-and-running" example of the audioset data.
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
balanced_train_h5 = h5py.File(str(AUDIOSET_SPLITS_V1 / 'unbalanced_train.h5'), 'r')
X_train = balanced_train_h5['X'][()]
Y_train = balanced_train_h5['y'][()].astype(int)

ID_array = balanced_train_h5['slugs'][()]

# first normalize all the data
X_train = X_train/255


# now split off last 5% of data to use as dev set
num_examples = Y_train.shape[0]
X_dev = X_train[ int(num_examples*0.95):,:,:,:]
X_train = X_train[ :int(num_examples*0.95)-1,:,:,:]
Y_dev = Y_train[ int(num_examples*0.95):,:]
Y_train = Y_train[ :int(num_examples*0.95)-1,:]




# dataset statistics:
n_examples_per_class= np.sum(Y_train, axis = 0, keepdims = True)
print("Total number of examples in train set:")
print(Y_train.shape[0])
print("Number of examples per class in train set: ")
print(n_examples_per_class)


print("Total number of examples in dev set:")
print(Y_dev.shape[0])
n_examples_per_class= np.sum(Y_dev, axis = 0, keepdims = True)
print("Number of examples per class in dev set: ")
print(n_examples_per_class)


#%% Save train and dev splits using pickle
pickle_out = open(str(AUDIOSET_SPLITS_V1 / 'train_and_dev_all_classes_v1.pickle'),"wb")
data = (X_train, Y_train, X_dev, Y_dev);

pkl.dump(data, pickle_out)

pickle_out.close()




