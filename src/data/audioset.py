#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Tools for interfacing with Google's audioset
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
import sqlite3
import tqdm
import h5py
load_dotenv()

AUDIOSET_PATH = os.getenv('AUDIOSET_PATH')
AUDIOSET_PATH = Path(AUDIOSET_PATH).expanduser()
AUDIOSET_SQLITE_DATABASE = os.getenv('AUDIOSET_SQLITE_DATABASE')
AUDIOSET_SQLITE_DATABASE = Path(AUDIOSET_SQLITE_DATABASE).expanduser()
AUDIOSET_H5_DATABASE = os.getenv('AUDIOSET_H5_DATABASE')
AUDIOSET_H5_DATABASE = Path(AUDIOSET_H5_DATABASE).expanduser()

EMBEDDINGS_PATH = AUDIOSET_PATH / 'vgg_features/audioset_v1_embeddings'
BAL_TRAIN_PATH = EMBEDDINGS_PATH / 'bal_train'
UNBAL_TRAIN_PATH = EMBEDDINGS_PATH / 'unbal_train'
EVAL_PATH = EMBEDDINGS_PATH / 'eval'
CLASS_LABELS_INDICES_PATH = AUDIOSET_PATH / 'class_labels_indices.csv'

class_labels_indices = pd.read_csv(CLASS_LABELS_INDICES_PATH)


def index(sqlite_conn, h5file):
    conn = sqlite_conn
    cursor = conn.cursor()

    # Drop the old tables (we assume we're remaking it...)
    cursor.execute('DROP TABLE IF EXISTS labels_videos;')
    cursor.execute('DROP TABLE IF EXISTS videos;')
    cursor.execute('DROP TABLE IF EXISTS labels;')

    # Create the tables
    with (Path(__file__).parent /'create_audioset_tables.sql').open('r') as f:
        sql = f.read()
    cursor.executescript(sql)

    # Populate the labels table
    class_labels_indices = pd.read_csv(CLASS_LABELS_INDICES_PATH)
    labels = [(i, row.mid, row.display_name)
              for i, row in class_labels_indices.iterrows()]
    sql = 'INSERT INTO labels VALUES(?, ?, ?)'
    cursor.executemany(sql, labels)
    conn.commit()

    # Populate the tables from the tfrecords
    tfrecord_filenames = (str(f) for f in BAL_TRAIN_PATH.glob('1*.tfrecord'))

    split_index = 0
    for tfrecord in tqdm.tqdm(list(tfrecord_filenames)):
        for example in tf.python_io.tf_record_iterator(tfrecord):
            tf_example = tf.train.Example.FromString(example)

            f = tf_example.features.feature
            video_id = (f['video_id'].bytes_list.value[0]
                        ).decode(encoding='UTF-8')

            start_time_seconds = f['start_time_seconds'].float_list.value[0]
            end_time_seconds = f['end_time_seconds'].float_list.value[0]

            label_ids = list(np.asarray(
                tf_example.features.feature['labels'].int64_list.value))

            tf_seq_example = tf.train.SequenceExample.FromString(example)
            fl = tf_seq_example.feature_lists.feature_list['audio_embedding']
            n_frames = len(fl.feature)

            with tf.Session() as sess:
                audio_frames = [tf.cast(tf.decode_raw(
                    fl.feature[i].bytes_list.value[0], tf.uint8), tf.uint8
                    ).eval()
                    for i in range(n_frames)]

            arr = np.array(audio_frames)
            h5file.create_dataset(
                name=video_id,
                data=arr)
            h5file.flush()


            sql = (
                'INSERT INTO videos'
                '(video_id)'
                'VALUES (?)'
                )
            cursor.execute(sql, (
                video_id,))
            conn.commit()

            sql = (
                'INSERT INTO labels_videos'
                '(video_id, label_id, split_id, start_time_seconds,'
                ' end_time_seconds)'
                'VALUES (?, ?, ?, ?, ?)'
                )

            params = tuple((video_id, int(label_id), split_index,
                            start_time_seconds, end_time_seconds)
                           for label_id in label_ids)
            cursor.executemany(sql, params)
    conn.commit()

class AudiosetManager():

    def __init__(self):
        self.h5file = h5py.File(str(AUDIOSET_H5_DATABASE), 'r')
        with sqlite3.connect(str(AUDIOSET_SQLITE_DATABASE)) as conn:
            sql = 'SELECT * FROM labels;'
            self._classes = pd.read_sql_query(sql, conn)

            sql = 'SELECT * FROM videos;'
            self._videos = pd.read_sql_query(sql, conn)

            sql = 'SELECT * from labels_videos;'
            self._labels_videos = pd.read_sql_query(sql, conn)

    @property
    def classes(self):
        return self._classes

    @property
    def videos(self):
        return self._videos

    @property
    def labels_videos(self):
        return self._labels_videos

    def get_vggish(self, key):
        return self.h5file[key][()]


if __name__ == '__main__':
    with sqlite3.connect(str(AUDIOSET_SQLITE_DATABASE)) as conn:
        with h5py.File(str(AUDIOSET_H5_DATABASE), 'w') as h5file:
            index(conn, h5file)

