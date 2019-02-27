import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
import sqlite3
import tqdm
import h5py
import logging
import operator
import sys
from functools import reduce
load_dotenv()


AUDIOSET_PATH = os.getenv('AUDIOSET_PATH')
AUDIOSET_PATH = Path(AUDIOSET_PATH).expanduser()
AUDIOSET_SQLITE_DATABASE = os.getenv('AUDIOSET_SQLITE_DATABASE')
AUDIOSET_SQLITE_DATABASE = Path(AUDIOSET_SQLITE_DATABASE).expanduser()
AUDIOSET_H5_DATABASE = os.getenv('AUDIOSET_H5_DATABASE')
AUDIOSET_H5_DATABASE = Path(AUDIOSET_H5_DATABASE).expanduser()

AUDIOSET_SPLITS_V1 = os.getenv('AUDIOSET_SPLITS_V1')
AUDIOSET_SPLITS_V1 = Path(AUDIOSET_SPLITS_V1).expanduser()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with sqlite3.connect(str(AUDIOSET_SQLITE_DATABASE)) as conn:
        split_index = 2
                      # 0 for balanced train
                      # 1 for eval
                      # 2 for unbalanced train

        duration = 10 # seconds
        number_of_features = 128
        number_of_labels = 1

        sql = """
        SELECT video_id, label_id FROM labels_videos
        WHERE ROUND(end_time_seconds - start_time_seconds, 2) = {duration}
        AND labels_videos.split_id = {split_index}
        AND label_id IN (
            SELECT
                --labels.display_name as label_name,
                --COUNT(labels_videos.id) as frequency,
                labels.id as label_id
            FROM labels_videos
            INNER JOIN labels ON labels.id = labels_videos.label_id
            WHERE labels_videos.split_id = {split_index} -- balanced train
            GROUP BY label_id
            ORDER BY COUNT(labels_videos.id) DESC
            LIMIT {number_of_labels}
        );
        """.format(
            split_index=split_index,
            duration=duration,
            number_of_labels=number_of_labels)

        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        all_set = {}
        logging.info('Building dictionary')
        tbar = tqdm.tqdm(results)
        for slug, label in tbar:
            if slug in all_set.keys():
                all_set[slug].append(label)
            else:
                all_set[slug] = [label]

    np.random.seed(42)
    keys = list(all_set.keys())
    logging.info('Counting labels')
    labels = np.unique(reduce(operator.add, all_set.values()))
    number_of_labels = 527
    if len(labels) != number_of_labels:
        logging.warning('Found %i labels out of %i' % (len(labels), number_of_labels))
    number_of_features = 128

    logging.info('Shuffling')
    np.random.shuffle(keys)
    train_keys = keys[:]  # Use the whole training set

    logging.info('Allocating memory')
    m = len(train_keys)
    X = np.zeros(shape=(m, duration, number_of_features, 1), dtype=np.uint8)
    y = np.zeros((m, number_of_labels), dtype=bool)
    slugs = np.empty(shape=(m), dtype='S11')  # 11-character unicode strings

    raw_h5file = h5py.File(str(AUDIOSET_H5_DATABASE), 'r')

    tbar = tqdm.tqdm(train_keys)
    for i, key in enumerate(tbar):
        tbar.set_description(key)
        data = raw_h5file[key][()]
        X[i, :, :, 0] = raw_h5file[key][()]
        for label in all_set[key]:
            y[i][label] = 1
        slugs[i] = key.encode('ascii')

    raw_h5file.close()

    AUDIOSET_SPLITS_V1.mkdir(parents=True, exist_ok=True)
    balanced_train_h5file = h5py.File(
        str(AUDIOSET_SPLITS_V1 / 'test.h5'), 'w')


    logging.info('Writing to disk')
    balanced_train_h5file.create_dataset(name='X', data=X)
    balanced_train_h5file.create_dataset(name='y', data=y)
    balanced_train_h5file.create_dataset(name='slugs', data=slugs, dtype='S11')
    logging.info('Complete')
    balanced_train_h5file.close()

#asciiList = [n.encode("ascii", "ignore") for n in strList]
#h5File.create_dataset('xxx', (len(asciiList),1),'S10', asciiList)

    # Save these for unbalanced train
    #test_keys = keys[0:5000]
    #validate_keys = keys[5000:10000]
    #train_keys = keys[10000:]
