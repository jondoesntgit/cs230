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

load_dotenv()

AUDIOSET_SPLITS_V1 = os.getenv('AUDIOSET_SPLITS_V1')
AUDIOSET_SPLITS_V1 = Path(AUDIOSET_SPLITS_V1).expanduser()


#%%
def customLoss(yTrue, yPred):
    # clip to avoid numerical overflow issues.
    yPred = keras.backend.clip(yPred, keras.backend.epsilon(), 1.0-keras.backend.epsilon())
    POSTIVE_WEIGHT_FACTOR = 1
    loss = -(POSTIVE_WEIGHT_FACTOR * yTrue * keras.backend.log(yPred) + (1-yTrue)*keras.backend.log(1-yPred))
    return keras.backend.mean(loss, axis=-1)

#%%

if __name__ == '__main__':

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
    
    classes_to_keep = [ 137, 1, 2, 111, 288, 343, 396]
    N_classes = len(classes_to_keep)
    #num_labels = len(Y_train[1])
    

    
    # keep only the data belonging to the classes above, with exactly 1 class label
    Y_train_reduced = Y_train[:,classes_to_keep]
    n_classes_per_example = np.sum(Y_train_reduced,axis=1,keepdims=True)
    single_class_examples = np.nonzero(n_classes_per_example == 1)
    Y_train_reduced = Y_train_reduced[single_class_examples[0],:]
    X_train_reduced = X_train[single_class_examples[0],:,:,:]
    

    # iterate through and remove most of the music examples
    N_music_to_keep = 20000
    music_index = 0 # new index in the classes_to_keep array
    integer_classes = Y_train_reduced.argmax(axis=1)
    music_indices = np.nonzero(integer_classes == music_index)
    music_indices = music_indices[0]
    non_music_indices = np.nonzero(integer_classes != music_index)
    non_music_indices = non_music_indices[0]
    new_indices = np.concatenate((music_indices[0:N_music_to_keep], non_music_indices))
    shuffle(new_indices)
    X_train_reduced = X_train_reduced[ new_indices ,:,:,: ] 
    Y_train_reduced = Y_train_reduced[ new_indices ,: ]     
    
    
    # create an "other" class set and add it in
    
    # these are the examples which don't contain any of our target labels
    non_target_examples = np.nonzero(n_classes_per_example == 0)[0]
    N_other_examples_to_keep = 20000

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
    
    
    #%%
    input_h = X_train.shape[1]
    input_w = X_train.shape[2]
    
    
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
    N_epochs = 10

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
    model.save('audioset_softmax_v02_dropout_0p5.h5')




