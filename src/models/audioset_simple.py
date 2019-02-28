#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
A simple "get-up-and-running" example of the audioset data.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from dotenv import load_dotenv
import h5py
import os
from pathlib import Path
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

    balanced_train_h5 = h5py.File(str(AUDIOSET_SPLITS_V1 / 'unbalanced_train.h5'), 'r')
    X_train = balanced_train_h5['X'][()]
    y_train = balanced_train_h5['y'][()].astype(int)
    
    
    
    num_labels = len(y_train[1])

    non_zero_labels = [
        i for i, val in enumerate(np.sum(y_train, axis=0)) if val]
    y_compressed = y_train[:, non_zero_labels]

    input_h = X_train.shape[1]
    input_w = X_train.shape[2]
    
    keras.backend.clear_session()
    model = tf.keras.Sequential()
    
    N_hidden_layers = 3
    N_dense = int(2e3)
    lr = .001
    N_epochs = 4
    minibatch_size = 32
    N_filters = 256
    
    # model 1: dense layers only
#    model.add(tf.keras.layers.Flatten(input_shape=(input_h, input_w, 1)))
#    for i in range(N_hidden_layers):
#        model.add(tf.keras.layers.Dense(N_dense, activation='relu', kernel_initializer = keras.initializers.he_uniform(seed=None)))
#    #model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
#    #model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
#    model.add(tf.keras.layers.Dense(len(non_zero_labels), activation='sigmoid', kernel_initializer = keras.initializers.he_uniform(seed=None)))
    
    
    
    # model 2: Conv layer followed by dense layers
    model.add(tf.keras.layers.Conv2D(
            filters=N_filters,
        kernel_size=(input_h, 3),
        padding='valid',
        strides=1,
        activation='relu',
        input_shape=(input_h, input_w, 1)))
    model.add(tf.keras.layers.Flatten())
    for i in range(N_hidden_layers):
        model.add(tf.keras.layers.Dense(N_dense, activation='relu', kernel_initializer = keras.initializers.he_uniform(seed=None)))
    #model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    #model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(len(non_zero_labels), activation='sigmoid', kernel_initializer = keras.initializers.he_uniform(seed=None)))
    
    
    
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=customLoss,
                  metrics=['accuracy'])
    
#    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
    
    #%%
    model.fit(X_train, y_compressed, epochs=N_epochs, batch_size=minibatch_size)
    
    #%% evaluate performance
    eval_h5 = h5py.File(str(AUDIOSET_SPLITS_V1 / 'eval.h5'), 'r')
    X_test = eval_h5['X'][()]
    y_test = eval_h5['y'][()].astype(int)
    y_test_compressed = y_test[:, non_zero_labels]
    
    test_loss, test_acc = model.evaluate(X_test, y_test_compressed)
    print('Test accuracy:', test_acc)
    
    #%%
    predictions = model.predict(X_test)
    prediction_summary = np.sum(predictions, axis=0,keepdims=True)
    print('Predicted class distribution:')
    print(prediction_summary)
    print('Number of classes predicted: ' + str(np.sum(prediction_summary >= 1)))
    
    target_summary = np.sum(y_test_compressed, axis=0,keepdims=True)
    print('Actual class distribution')
    print(target_summary)
    
    #%%
    #    model.save('audioset_simple.h5')
