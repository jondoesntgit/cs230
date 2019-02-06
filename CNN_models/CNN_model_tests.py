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
import matplotlib.pyplot as plt
print(tf.__version__)


#%% Load dataset

# here, load built in fasion mnist dataset rather than spectrograms as a test
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



#%% Do some dataset pre-processing, if necessary.

train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape (add a dimenion) for the CNN
train_images = train_images.reshape(train_images.shape[0],train_images.shape[1],train_images.shape[2],1)
test_images = test_images.reshape(test_images.shape[0],test_images.shape[1],test_images.shape[2],1)


#%% Do some helpful plots to visualize dataset.
#
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#%% build Keras model with CNNs

# clear any old models/layers
keras.backend.clear_session() 


# model 1, do CNN for MNIST fashion, described here: 
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

model1 = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.Flatten(),
    #keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model1.summary()

# For audio, would like to use model similar to the one in this paper: 
# http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf

# actual input shape would be roughly 128 x 430, but try this for now
# note: for specifying kernel size, tuple is (height,width)
input_h = 28
input_w = 28

N_filters = 80
N_dense = 1000

N_classes = 10
model2 = tf.keras.Sequential()

model2.add(tf.keras.layers.Conv2D(filters=N_filters, kernel_size=(input_h - 3,6), padding='valid', strides = 1, activation='relu', input_shape=(input_h,input_w,1))) 
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(4,3), strides=(4,3)))
model2.add(tf.keras.layers.Dropout(0.5))
model2.add(tf.keras.layers.Conv2D(filters=N_filters, kernel_size=(1,3), padding='valid', activation='relu'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3), strides=(1,3)))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(N_dense, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.5))
model2.add(tf.keras.layers.Dense(N_dense, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.5))
model2.add(tf.keras.layers.Dense(N_classes, activation='softmax'))

# Take a look at the model summary
model2.summary()




#%%
model2.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    
    
    
#%%
# takes ~13 mins to train, achieves 94.8% train accuracy and 91.4% test accuracy.
model2.fit(train_images, train_labels, epochs=5)

#%%

test_loss, test_acc = model2.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)





