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

    # Build a one-hot matrix (with multiple "one-hots")
    df = am.get_data_with_single_label()

    # Filter out short clips (6 seconds, etc...)
    keys = [key for key in df.video_id.values
            if am.get_vggish(key).shape[0] == 10]

    num_labels = max(df.label_id.values) + 1
    y_train = np.zeros((len(keys), num_labels))
    for i, key in enumerate(keys):
        ids = df.query(f'video_id == "{key}"').label_id.values
        for l in ids:
            y_train[i][l] = 1

    X_train = np.stack([am.get_vggish(key) for key in keys], axis=0)
    X_train = np.expand_dims(X_train, axis=-1)
    print(X_train.shape)

    input_h = X_train.shape[1]
    input_w = X_train.shape[2]

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
    model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=N_epochs, batch_size=minibatch_size)
