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



# For audio, would like to use model similar to the one in this paper: 
# http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf





#%%
model1.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    
    
    
#%%
# takes ~13 mins to train, achieves 94.8% train accuracy and 91.4% test accuracy.
model1.fit(train_images, train_labels, epochs=5)

#%%

test_loss, test_acc = model1.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)



