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

# TODO:
# This doesn't quite run yet, but most of the parts are here. Tensorflow is
# outputting some helpful error messages, but I don't have time to make it work
# at the moment.

if __name__ == '__main__':

    balanced_train_h5 = h5py.File(str(AUDIOSET_SPLITS_V1 / 'balanced_top_5.h5'), 'r')
    X_train = balanced_train_h5['X'][()]
    y_train = balanced_train_h5['y'][()].astype(int)

    num_labels = len(y_train[1])
    input_h = X_train.shape[1]
    input_w = X_train.shape[2]

    keras.backend.clear_session()
    model = tf.keras.Sequential()

    N_dense = int(5e2)
    lr = .001
    N_epochs = 10
    minibatch_size = 32

    model.add(tf.keras.layers.Flatten(input_shape=(input_h, input_w, 1)))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(num_labels, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=N_epochs, batch_size=minibatch_size)

    model.save('audioset_simple.h5')
