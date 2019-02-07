#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
A simple "get-up-and-running" example of the audioset data.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
sys.path.append('..')
from data.audioset import AudiosetManager

# TODO:
# This doesn't quite run yet, but most of the parts are here. Tensorflow is
# outputting some helpful error messages, but I don't have time to make it work
# at the moment.

if __name__ == '__main__':
    am = AudiosetManager()

    # Filter out short clips (6 seconds, etc...)
    keys = [key for key in am.videos.video_id.values
            if am.get_vggish(key).shape[0] == 10]

    # Build a one-hot matrix (with multiple "one-hots")
    df = am.labels_videos
    num_labels = max(df.label_id.values) + 1
    y_train = np.zeros((len(keys), num_labels))
    for i, key in enumerate(keys):
        ids = df.query(f'video_id == "{key}"').label_id.values
        for l in ids:
            y_train[i][l] = 1

    X_train = np.stack([am.get_vggish(key) for key in keys], axis=0)
    X_train = np.expand_dims(X_train, axis=-1)

    input_h = X_train.shape[1]
    input_w = X_train.shape[2]
    N_classes = np.max(y_train) + 1

    keras.backend.clear_session()
    model = tf.keras.Sequential()

    N_dense = 500
    lr = .001
    N_epochs = 5
    minibatch_size=32

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(N_dense, activation='relu'))
    model.add(tf.keras.layers.Dense(N_classes, activation='softmax'))

    adamOptimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adamOptimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=N_epochs, batch_size=minibatch_size)
