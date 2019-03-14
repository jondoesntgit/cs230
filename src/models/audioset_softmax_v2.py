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

if __name__ == '__main__':

    #%% load data numpy files
    pickle_in = open(str(AUDIOSET_SPLITS_V1 / 'train_and_dev_7_classes_v2.pickle'), 'rb')
    data = pkl.load(pickle_in)
    (X_train_reduced, Y_train_reduced, X_dev, Y_dev) = data
    
    input_h = X_train_reduced.shape[1]
    input_w = X_train_reduced.shape[2]
    N_classes = Y_train_reduced.shape[1]
    
    n_examples_per_class= np.sum(Y_train_reduced, axis = 0, keepdims = True)
    print('=========================================================')
    print('DATA SET STATISTICS')
    print('---------------------------------------------------------')
    print("Total number of examples in train set:")
    print(Y_train_reduced.shape[0])
    print("Number of examples per class in train set: ")
    print(n_examples_per_class)
    
    
    print("Total number of examples in dev set:")
    print(Y_dev.shape[0])
    n_examples_per_class= np.sum(Y_dev, axis = 0, keepdims = True)
    print("Number of examples per class in dev set: ")
    print(n_examples_per_class)
    print('=========================================================')

    
    #%% parameters
    N_hidden_layers = 3
    N_dense = int(1e3)
    lr = 1e-4
    minibatch_size = 32
    N_filters = 256
    drop_prob = 0.5
    seed = 1
    
    
    #%%
    K.clear_session()

    
    
#%% model 1: dense layers only
#    model = tf.keras.Sequential()
#    model.add(layers.Flatten(input_shape=(input_h, input_w, 1)))
#    for i in range(N_hidden_layers):
#        model.add(layers.Dense(N_dense, activation='relu', kernel_initializer = keras.initializers.he_uniform(seed=None)))
#    #model.add(layers.Dense(N_dense, activation='relu'))
#    #model.add(layers.Dense(N_dense, activation='relu'))
#    model.add(layers.Dense(N_classes, activation='softmax', kernel_initializer = keras.initializers.he_uniform(seed=None)))
#    
    
    
    #%% model 2: Conv layer followed by dense layers
#    model = tf.keras.Sequential()
#    model.add(layers.Conv2D(
#            filters=N_filters,
#        kernel_size=(3, input_w),
#        padding='valid',
#        strides=1,
#        activation='relu',
#        input_shape=(input_h, input_w, 1)))
#    model.add(layers.Flatten())
#    for i in range(N_hidden_layers):
#        model.add(layers.Dense(N_dense, activation='relu', kernel_initializer = keras.initializers.he_uniform(seed=None)))
#    model.add(layers.Dense(N_classes, activation='softmax', kernel_initializer = keras.initializers.he_uniform(seed=None)))
#    
#    
    #%% model 3: includes Conv layer, not made sequentially
    input_shape=(input_h, input_w, 1)
    X_input = Input(shape = input_shape)
    X = X_input
    X = Conv2D(filters=N_filters,
        kernel_size=(3, input_w),
        padding='valid',
        strides=1,
        activation='relu',
        kernel_initializer = keras.initializers.he_uniform(seed=seed))(X)
    X = Flatten()(X)
    for i in range(N_hidden_layers):
        X = Dense(N_dense, 
                  activation='relu',
                  kernel_initializer = keras.initializers.he_uniform(seed=seed))(X) 
        X = Dropout(drop_prob)(X)
    X = Dense(N_classes, 
              activation=None,
              kernel_initializer = keras.initializers.he_uniform(seed=seed))(X)
    output = Activation('softmax')(X)
    model = Model(inputs = X_input, outputs = output)
    
    #%%    
    
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
#    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
    
    #%%
    N_epochs = 3

    for i in range(N_epochs):
        print('Epoch: ' + str(i+1))
        model.fit(X_train_reduced, Y_train_reduced, epochs=1, batch_size=32)
        test_loss, test_acc = model.evaluate(X_dev, Y_dev)
        print('Test accuracy:', test_acc)
        
        
    
    
    #%% evaluate layer outputs after fitting
#    inp = model.input                                           # input placeholder
#    outputs = []
#    for i in range(1,len(model.layers)):
#        layer = model.layers[i]
#        outputs.append(layer.output)
#    functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function
#    
#    # Testing
#    for i in range(X_train_micro.shape[0]):
##    i = 1
#        test_input = X_train_micro[i,:,:,:]
#        layer_outs = functor([test_input, 1.])
#        print(layer_outs[len(layer_outs)-1])
#    
    
    
    #%% evaluate performance
#    eval_h5 = h5py.File(str(AUDIOSET_SPLITS_V1 / 'eval.h5'), 'r')
#    X_test = eval_h5['X'][()]
#    y_test = eval_h5['y'][()].astype(int)
#    y_test_compressed = y_test[:, non_zero_labels]
    
    test_loss, test_acc = model.evaluate(X_dev, Y_dev)
    print('Test accuracy:', test_acc)
    
    #%%
    predictions = model.predict(X_dev)
    prediction_summary = np.sum(predictions, axis=0,keepdims=True)
    print('Predicted class distribution:')
    print(prediction_summary)
    print('Number of classes predicted: ' + str(np.sum(prediction_summary >= 1)))
    
    target_summary = np.sum(Y_dev, axis=0,keepdims=True)
    print('Actual class distribution')
    print(target_summary)
    
    
    #%% create confusion matrix for dev set
    
    dev_confusion_matrix = metrics.confusion_matrix(Y_dev.argmax(axis=1), predictions.argmax(axis=1))
    print('Dev set confusion matrix:')
    print(dev_confusion_matrix)
    
    #%%
#   model.save('audioset_softmax_run46.h5')




