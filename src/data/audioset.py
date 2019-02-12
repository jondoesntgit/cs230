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
import logging
load_dotenv()

AUDIOSET_PATH = os.getenv('AUDIOSET_PATH')
AUDIOSET_PATH = Path(AUDIOSET_PATH).expanduser()
AUDIOSET_SQLITE_DATABASE = os.getenv('AUDIOSET_SQLITE_DATABASE')
AUDIOSET_SQLITE_DATABASE = Path(AUDIOSET_SQLITE_DATABASE).expanduser()
AUDIOSET_H5_DATABASE = os.getenv('AUDIOSET_H5_DATABASE')
AUDIOSET_H5_DATABASE = Path(AUDIOSET_H5_DATABASE).expanduser()

EMBEDDINGS_PATH = AUDIOSET_PATH / 'audioset_v1_embeddings'
BAL_TRAIN_PATH = EMBEDDINGS_PATH / 'bal_train'
UNBAL_TRAIN_PATH = EMBEDDINGS_PATH / 'unbal_train'
EVAL_PATH = EMBEDDINGS_PATH / 'eval'
CLASS_LABELS_INDICES_PATH = AUDIOSET_PATH / 'class_labels_indices.csv'

class_labels_indices = pd.read_csv(CLASS_LABELS_INDICES_PATH)


def index_tfrecord(conn, h5file, tfrecord, split_index):
    cursor = conn.cursor()
    ebar = tqdm.tqdm(list(tf.python_io.tf_record_iterator(tfrecord)))
    for example in ebar:
        tf_example = tf.train.Example.FromString(example)

        f = tf_example.features.feature
        video_id = f['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
        ebar.set_description('youtube.com/%s' % video_id)

        # TODO: This is not failproof. This may skip videos that have
        # multiple labels
        sql = 'SELECT COUNT(*) FROM labels_videos WHERE video_id = ?'
        cursor.execute(sql, (video_id,))
        num_rows = cursor.fetchone()[0]
        if num_rows:
            continue

        if len(f['start_time_seconds'].float_list.value) > 1:
            print(video_id)
        start_time_seconds = f['start_time_seconds'].float_list.value[0]
        end_time_seconds = f['end_time_seconds'].float_list.value[0]

        label_ids = list(np.asarray(
            tf_example.features.feature['labels'].int64_list.value))

        tf_seq_example = tf.train.SequenceExample.FromString(example)
        fl = tf_seq_example.feature_lists.feature_list['audio_embedding']
        n_frames = len(fl.feature)

        length = 2 * 128

        if video_id not in h5file.keys():
            arr = np.array([
            [int(hex_embed[i:i+2], 16) 
                for i in range(0, length, 2)]
            for hex_embed in [
                fl.feature[frame_index].bytes_list.value[0].hex() 
                for frame_index in range(n_frames)
                ]])
            h5file.create_dataset(
                name=video_id,
                data=arr)
            h5file.flush()
        else:
            logging.warning('%s has already been indexed' % video_id)

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

def index_folder(conn, h5file, folder, split_index):
    tfrecord_filenames = list(folder.glob('*.tfrecord'))
    tbar = tqdm.tqdm(tfrecord_filenames)
    for tfrecord in tbar:
        tbar.set_description('Indexing %s' % tfrecord.name)
        index_tfrecord(conn, h5file, str(tfrecord), split_index)
    conn.commit()

def index_all(conn, h5file):
    for split_index, folder in enumerate(
            [BAL_TRAIN_PATH, EVAL_PATH, UNBAL_TRAIN_PATH]):
        logging.info('Indexing %s' % folder.name)
        index_folder(conn, h5file, folder, split_index)
        conn.commit()


def make_tables(conn, h5file, force=False):
    cursor = conn.cursor()

    # Drop the old tables (we assume we're remaking it...)
    if force:
        logging.info('Dropping tables')
        cursor.execute('DROP TABLE IF EXISTS labels_videos;')
        cursor.execute('DROP TABLE IF EXISTS videos;')
        cursor.execute('DROP TABLE IF EXISTS labels;')

        # Create the tables
        sql_path = (Path(__file__).parent/'create_audioset_tables.sql')
        with sql_path.open('r') as f:
            sql = f.read()
        cursor.executescript(sql)

        # Populate the labels table
        class_labels_indices = pd.read_csv(CLASS_LABELS_INDICES_PATH)
        labels = [(i, row.mid, row.display_name)
                  for i, row in class_labels_indices.iterrows()]
        sql = 'INSERT INTO labels VALUES(?, ?, ?)'
        cursor.executemany(sql, labels)
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

    def get_data_with_single_label(self, labels=None):
        with sqlite3.connect(str(AUDIOSET_SQLITE_DATABASE)) as conn:
            sql = (
                'SELECT * FROM labels_videos '
                'GROUP BY video_id '
                'HAVING COUNT(*)=1 '
                )
            df = pd.read_sql_query(sql, conn)
            if labels is not None:
                return df[df.label_id.isin(labels)]
            return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with sqlite3.connect(str(AUDIOSET_SQLITE_DATABASE)) as conn:
        with h5py.File(str(AUDIOSET_H5_DATABASE), 'w') as h5file:
            make_tables(conn, h5file, force=True)
            index_all(conn, h5file)
