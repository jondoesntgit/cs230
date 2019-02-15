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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with sqlite3.connect(str(AUDIOSET_SQLITE_DATABASE)) as conn:
        sql = """
        SELECT video_id, label_id FROM labels_videos
        WHERE ROUND(end_time_seconds - start_time_seconds, 2) = 10
        AND labels_videos.split_id = 2
        AND label_id IN (
            SELECT
                --labels.display_name as label_name,
                --COUNT(labels_videos.id) as frequency,
                labels.id as label_id
            FROM labels_videos
            INNER JOIN labels ON labels.id = labels_videos.label_id
            WHERE labels_videos.split_id = 2 -- unbalanced train
            GROUP BY label_id
            ORDER BY COUNT(labels_videos.id) DESC
            LIMIT 40
        );
        """

        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        all_set = {}
        for slug, label in results:
            if slug in all_set.keys():
                all_set[slug].append(label)
            else:
                all_set[slug] = [label]
        np.random.seed(42)
        keys = list(all_set.keys())
        np.random.shuffle(keys)
        test_keys = keys[0:5000]
        validate_keys = keys[5000:10000]
        train_keys = keys[10000:]

        # TODO: write them to an h5py file.
