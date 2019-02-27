#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from Jeremy's PiczakModel.py to process the esc50 splits.
"""

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import logging
import sys
import getopt

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logging.debug("Tensorflow version: %s" % tf.__version__)


def run_piczak_model(X_train, y_train, X_test, y_test, lr=0.001, N_epochs=5,
                     N_filters=80, N_dense=500, keep_prob=0.5,
                     minibatch_size=32):
    # extract dimensions of data

    input_h = X_train.shape[1]
    input_w = X_train.shape[2]
    N_classes = np.max(y_train) + 1

    # clear any old models/layers
    keras.backend.clear_session()

    # build Piczak model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=N_filters,
        kernel_size=(input_h - 3, 6),
        padding='valid',
        strides=1,
        activation='relu',
        input_shape=(input_h, input_w, 1)
        ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 3), strides=(4, 3)))
    model.add(tf.keras.layers.Dropout(1 - keep_prob))
    model.add(tf.keras.layers.Conv2D(filters=N_filters, kernel_size=(1, 3),
                                     padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dropout(1 - keep_prob))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dropout(1 - keep_prob))
    model.add(tf.keras.layers.Dense(N_classes, activation='softmax'))

    # print the model summary
    model.summary()

    # create optimizer
    adam_optimizer = tf.keras.optimizers.Adam(lr=lr)

    # compile model
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # run model
    model.fit(X_train, y_train, epochs=N_epochs, batch_size=minibatch_size)

    # evaluate performance
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # return accuracy and model
    return test_acc, model


def main():
    """
    Run a test of functionality by loading the esc50 dataset and run the
    standard Piczak model.
    """

    options, remainder = getopt.gnu_getopt(sys.argv[1:], 'e', ['epochs='])
    epochs = 50
    for opt, arg in options:
        if opt in ('-e', '--epochs'):
            if arg != "":
                epochs = int(arg)

    ESC50_SPLITS_PATH = Path(os.getenv('ESC50_SPLITS')).expanduser()

    X_train = np.load(ESC50_SPLITS_PATH / 'train/train_data.npy')
    y_train = np.load(ESC50_SPLITS_PATH / 'train/train_label.npy')

    X_test = np.load(ESC50_SPLITS_PATH / 'test/test_data.npy')
    y_test = np.load(ESC50_SPLITS_PATH / 'test/test_label.npy')

    # reshape (to add a dimension) for the CNN

    X_train = X_train.reshape(*X_train.shape, 1)
    X_train = np.swapaxes(X_train, 0, 2)

    X_test = X_test.reshape(*X_test.shape, 1)
    X_test = np.swapaxes(X_test, 0, 2)

    logging.debug(f'X_train shape: {X_train.shape}')
    logging.debug(f'y_train shape: {y_train.shape}')
    logging.debug(f'X_test shape: {X_test.shape}')
    logging.debug(f'y_test shape: {y_test.shape}')

    accuracy, model = run_piczak_model(X_train, y_train, X_test, y_test, 
        N_epochs=epochs, lr=0.001, N_filters=80, N_dense=2000, 
        keep_prob=0.3, minibatch_size=32)

    model.save('piczak.h5')


if __name__ == '__main__':
    main()
