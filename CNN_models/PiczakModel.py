#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:23:14 2019

@author: jwitmer
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras 

# Helper libraries
import numpy as np
print(tf.__version__)


#%%
def runPiczakModel(X_train, Y_train, X_test, Y_test, lr=0.001, N_epochs=5, 
                   N_filters=80, N_dense=500, keep_prob=0.5, minibatch_size=32):
# inputs: 
#         X_train: a numpy array of shape (batch_size, input_h, input_w, 1)
#         Y_train: a numpy array of shape (batch_size,1).  This should use sparse
#            encoding, ie. numbers from 0 to N_classes-1 (eg. 0 to 9 for 10 classes)
#            NOT one-hot coding
#         X_test: a numpy array of shape (batch_size, input_h, input_w, 1)
#         Y_test: a numpy array of shape (batch_size,1)
#         lr: learning rate
#         N_epochs: number of training epochs
#         N_filters: number of filters used in each the two Conv2D layers.
#         N_dense: number of nodes in each of the two fully connected layers.
#         keep_prob: keep_prob used for each of the three dropout steps. 
#         minibatch_size: mini-batch size used in gradient descent   
#
# outputs:
#        accuracy: the classification accuracy on the test set
#        model: the trained model
#
# for more information see paper: 
# https://ieeexplore.ieee.org/abstract/document/7324337


    # extract relevant dimensions of the data:
    input_h = X_train.shape[1]
    input_w = X_train.shape[2]    
    N_classes = np.max(Y_train) + 1
        
    # clear any old models/layers
    keras.backend.clear_session()
    
    # build Piczak model 
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(filters=N_filters, kernel_size=(input_h - 3,6), 
                                     padding='valid', strides = 1, activation='relu', 
                                     input_shape=(input_h,input_w,1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4,3), strides=(4,3)))
    model.add(tf.keras.layers.Dropout(1 - keep_prob))
    model.add(tf.keras.layers.Conv2D(filters=N_filters, kernel_size=(1,3),
                                     padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3), strides=(1,3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dropout(1 - keep_prob))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dropout(1 - keep_prob))
    model.add(tf.keras.layers.Dense(N_classes, activation='softmax'))
    
    # Print the model summary
    model.summary()

    # create optimizer
    adamOptimizer = tf.keras.optimizers.Adam(lr=lr)
    
    # compile model
    model.compile(optimizer=adamOptimizer, 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])    
        
    # run model  
    model.fit(X_train, Y_train, epochs=N_epochs, batch_size=minibatch_size)
    
    # evaluate performance
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)
    
    # return accuracy and model
    return test_acc, model




#%%
def main():
    # do a test of functionality by loading fashion_mnist dataset and running 
    # the standard Piczak model
    
    # Load dataset
    
    # here, load built in fasion mnist dataset rather than spectrograms as a test
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    
    #Do some dataset pre-processing
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # reshape (add a dimenion) for the CNN
    train_images = train_images.reshape(train_images.shape[0],train_images.shape[1],train_images.shape[2],1)
    test_images = test_images.reshape(test_images.shape[0],test_images.shape[1],test_images.shape[2],1)

    # call the Piczak model
    accuracy, model = runPiczakModel(train_images, train_labels, test_images, test_labels)



#%%
if __name__ == '__main__':
    main()


