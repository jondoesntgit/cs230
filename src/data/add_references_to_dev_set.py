#!/usr/bin/env python

from optparse import OptionParser
from pathlib import Path
from dotenv import load_dotenv
import os
import tensorflow as tf
import h5py
import numpy as np
import tqdm
import psycopg2
import psycopg2.extras
import tempfile
import boto3

load_dotenv()

def get_arr_from_slug(slug, tfrecords_dir):
    i = 0
    for t in tfrecords_dir.glob('bal_train/%s.tfrecord' % (slug[0:2], )):
        examples = tf.python_io.tf_record_iterator(str(t))
        for example in examples:
            i += 1
            tf_example = tf.train.Example.FromString(example)
            f = tf_example.features.feature
            video_id = f['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
            if slug != video_id:
                continue
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            fl = tf_seq_example.feature_lists.feature_list['audio_embedding']
            n_frames = len(fl.feature)

            length = 2 * 128
            print('Success')
            return np.array([
                    [int(hex_embed[i:i+2], 16)
                        for i in range(0, length, 2)]
                    for hex_embed in [
                        fl.feature[frame_index].bytes_list.value[0].hex()
                        for frame_index in range(n_frames)
                    ]])
    print(f'I could not find the example among {i} examples')

def add_slugs_to_work_list(slugs):
    PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
    PSQL_USERNAME = os.getenv('PSQL_USERNAME')
    PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
    PSQL_DATABASE = os.getenv('PSQL_DATABASE')

    conn = psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD)

    cur = conn.cursor()
    query = '''
    INSERT INTO embeddings (video_id, filter_id) values %s ON CONFLICT DO NOTHING;
    '''

    data = [(s, 1) for s in slugs]

    psycopg2.extras.execute_values(
        cur, query, data, template=None, page_size=100
    )
    conn.commit()

def fetch_by_slugs(slugs):
    PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
    PSQL_USERNAME = os.getenv('PSQL_USERNAME')
    PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
    PSQL_DATABASE = os.getenv('PSQL_DATABASE')

    conn = psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD)

    cur = conn.cursor()
    query = '''
    SELECT COUNT(*) from embeddings WHERE filter_id=1 and video_id IN %s AND aws_key IS NULL;
    '''

    cur.execute(query, (tuple(slugs),))
    rows = cur.fetchall()

    if int(rows[0][0]) > 0:
        print('Waiting on', rows[0][0], 'results')
        return
    
    query = '''
    SELECT video_id, aws_key from embeddings WHERE filter_id=1 and video_id IN %s;
    '''

    cur.execute(query, (tuple(slugs),))
    rows = cur.fetchall()
    lookup = {}
    for row in rows:
        video_id, aws_key = row
        lookup[video_id] = aws_key.strip()

    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = 'cs230-deep-audio'
    s3 = boto3.client(
    's3',
    # Hard coded strings as credentials, not recommended.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

    local_dir = tempfile.TemporaryDirectory()
    local_file = local_dir.name + '/arr.npy'
    original_arrays = []
    for slug in tqdm.tqdm(slugs):
        try:
            aws_key = lookup[slug]
        except:
            # Dataset changed while we were working. Fallback on filtered data
            aws_key = 'filter2/%s.npy' % slug
        s3.download_file(bucket_name, aws_key, local_file)
        arr = np.load(local_file)
        original_arrays.append(arr)
    X_orig = np.array(original_arrays)
    return np.swapaxes(X_orig, 1, 2)

def main(filename, tfrecords_dir):
    h5file = h5py.File(filename, mode='a')
    
    slugs = h5file['slugs'][:]
    slugs = [s.decode() for s in slugs]
    add_slugs_to_work_list(slugs)
    data = fetch_by_slugs(slugs)
    try:
        h5file['X_unfiltered'] = data
    except RuntimeError:
        del h5file['X_unfiltered']
        h5file['X_unfiltered'] = data


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--file', dest='filename', type='str', help='The .h5 filename that we are writing to')
    parser.add_option('-t', '--tfrecords_dir', dest='tfrecords_dir', type='str', help='The the directory that contains the tfrecords of interest')
    options, args = parser.parse_args()

    filename = Path(options.filename).expanduser()
    if options.tfrecords_dir:
        # Great, the user told us where to look
        tfrecords_dir = Path(options.tfrecords_dir).expanduser()
    else:
        # Uh oh, we have to guess now
 
        tfrecords_dir = os.environ['AUDIOSET_PATH']
        tfrecords_dir = Path(tfrecords_dir).expanduser()
        tfrecords_dir /= 'audioset_v1_embeddings'

    main(filename, tfrecords_dir)