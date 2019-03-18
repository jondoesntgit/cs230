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


# subselect N favourite classes
# male speech: 1
# female speech: 2
# bird: 111
# water: 288
# engine: 343
# siren: 396
# music: 137

#classes_to_keep = [ 137, 1, 2, 111, 288, 343, 396]  # this one includes music
classes_to_keep = [ 1, 2, 111, 288, 343, 396]
N_classes = len(classes_to_keep)
#num_labels = len(Y_train[1])



# keep only the data belonging to the classes above, with exactly 1 class label
Y_train_reduced = Y_train[:,classes_to_keep]
n_classes_per_example = np.sum(Y_train_reduced,axis=1,keepdims=True)
single_class_examples = np.nonzero(n_classes_per_example == 1)
Y_train_reduced = Y_train_reduced[single_class_examples[0],:]
X_train_reduced = X_train[single_class_examples[0],:,:,:]


# iterate through and remove most of the music examples
#N_music_to_keep = 20000
#music_index = 0 # new index in the classes_to_keep array
#integer_classes = Y_train_reduced.argmax(axis=1)
#music_indices = np.nonzero(integer_classes == music_index)
#music_indices = music_indices[0]
#non_music_indices = np.nonzero(integer_classes != music_index)
#non_music_indices = non_music_indices[0]
#new_indices = np.concatenate((music_indices[0:N_music_to_keep], non_music_indices))
#shuffle(new_indices)
#X_train_reduced = X_train_reduced[ new_indices ,:,:,: ] 
#Y_train_reduced = Y_train_reduced[ new_indices ,: ]     


## create an "other" class set and add it in

# these are the examples which don't contain any of our target labels
non_target_examples = np.nonzero(n_classes_per_example == 0)[0]
N_other_examples_to_keep = 23000

X_train_other = X_train[non_target_examples[0:N_other_examples_to_keep],:,:,:]
other_class_index = Y_train_reduced.shape[1]
Y_train_other = np.zeros([X_train_other.shape[0], other_class_index + 1])
Y_train_other[:,other_class_index] = 1

# now add this into X_train_reduced and Y_train_reduced
X_train_reduced = np.concatenate((X_train_reduced, X_train_other), axis=0 )
# add a new row to label matrix
Y_train_reduced = np.concatenate((Y_train_reduced, np.zeros([Y_train_reduced.shape[0],1]) ), 
                                 axis=1)
Y_train_reduced = np.concatenate((Y_train_reduced, Y_train_other), axis=0 )

# now shuffle the data
shuffled_indices = list(range(0,X_train_reduced.shape[0]))
shuffle(shuffled_indices)
X_train_reduced = X_train_reduced[shuffled_indices]
Y_train_reduced = Y_train_reduced[shuffled_indices]


#%%
# ONLY DO THIS IF YOU WANT SIGMOID ACTIVATIONS instead of explicit Other category.
Y_train_reduced = Y_train_reduced[:,:6]

#%%
# now split off last 5% of data to use as dev set
num_examples = Y_train_reduced.shape[0]
X_dev = X_train_reduced[ int(num_examples*0.95):,:,:,:]
X_train_reduced = X_train_reduced[ :int(num_examples*0.95)-1,:,:,:]
Y_dev = Y_train_reduced[ int(num_examples*0.95):,:]
Y_train_reduced = Y_train_reduced[ :int(num_examples*0.95)-1,:]



#    # make a mini train set using first 5% of data to see overfitting.
#    X_train_mini = X_train_reduced[ :int(num_examples*0.05),:,:,:]
#    Y_train_mini = Y_train_reduced[ :int(num_examples*0.05),:]
#   
#    # make a micro train set with just 2 examples per class
#    num_examples_per_class = 10
#    Y_train_micro = np.zeros([N_classes*num_examples_per_class,N_classes])
#    X_train_micro = np.zeros([N_classes*num_examples_per_class,X_train.shape[1], X_train.shape[2],X_train.shape[3]])
#    for i in range(N_classes):
#        class_examples = np.nonzero(Y_train_reduced[:,i])
#        for j in range(num_examples_per_class):
#            Y_train_micro[num_examples_per_class*i + j,:]= Y_train_reduced[class_examples[0][j],:]
#            X_train_micro[num_examples_per_class*i + j,:,:,:]= X_train_reduced[class_examples[0][j],:,:,:]
#
#    # normalize dataset
#    X_train_micro = X_train_micro / 255
#    X_norm = X_train_micro
#    mean = np.mean(X_train_micro)
#    var = np.var(X_train_micro)
#    X_norm = (X_train_micro - mean) / np.sqrt(var)



# dataset statistics:
n_examples_per_class= np.sum(Y_train_reduced, axis = 0, keepdims = True)
print("Total number of examples in train set:")
print(Y_train_reduced.shape[0])
print("Number of examples per class in train set: ")
print(n_examples_per_class)

#    print("Total number of examples in mini train set:")
#    print(Y_train_mini.shape[0])
#    n_examples_per_class= np.sum(Y_train_mini, axis = 0, keepdims = True)
#    print("Number of examples per class in mini train set: ")
#    print(n_examples_per_class)
#    
#    print("Total number of examples in micro train set:")
#    print(Y_train_micro.shape[0])
#    n_examples_per_class= np.sum(Y_train_micro, axis = 0, keepdims = True)
#    print("Number of examples per class in micro train set: ")
#    print(n_examples_per_class)
#    


print("Total number of examples in dev set:")
print(Y_dev.shape[0])
n_examples_per_class= np.sum(Y_dev, axis = 0, keepdims = True)
print("Number of examples per class in dev set: ")
print(n_examples_per_class)


#%% Save train and dev splits using pickle
pickle_out = open(str(AUDIOSET_SPLITS_V1 / 'train_and_dev_6_classes_plus_other_sigmoid_FINAL.pickle'),"wb")
data = (X_train_reduced, Y_train_reduced, X_dev, Y_dev);

pkl.dump(data, pickle_out)

pickle_out.close()




