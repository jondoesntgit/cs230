#!/usr/bin/env python

"""
Fetches a worklist from a databse and converts youtube slugs to vggish features
and stores them on AWS
"""
import sys
sys.path.append('..')
from features import vggish_slim, vggish_params, vggish_postprocess, mel_features
from features.vggish_input import wavfile_to_examples as w2e
from pathlib import Path
from dotenv import load_dotenv
import os
import boto3
import librosa as lib

import tensorflow as tf
import numpy as np

import psycopg2
import os
import dotenv
import requests
import json
import socket # to get hostname
import tempfile
import youtube_dl

dotenv.load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)

PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
PSQL_USERNAME = os.getenv('PSQL_USERNAME')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
PSQL_DATABASE = os.getenv('PSQL_DATABASE')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

CHECKPOINT = os.getenv('VGGISH_MODEL_CHECKPOINT')
CHECKPOINT = str(Path(CHECKPOINT).expanduser())
PCA_PARAMS = os.getenv('EMBEDDING_PCA_PARAMETERS')
PCA_PARAMS = str(Path(PCA_PARAMS).expanduser())

# This appears to be a fix for a common problem on macs...
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Hacky way to suppress a warning in vggish extraction


CREATE_TABLES = """

CREATE TYPE split AS ENUM ('train', 'dev', 'test');

CREATE TABLE videos (
    id CHARACTER(11) PRIMARY KEY,
    start_time integer,
    end_time integer,
    split split,
    UNIQUE (id, start_time, end_time)
);

CREATE TABLE labels (
    id CHARACTER(20) PRIMARY KEY,
    display_name CHARACTER(50) UNIQUE,
    parent_id CHARACTER(20) REFERENCES labels (id)
);

CREATE TABLE labels_videos (
    id serial PRIMARY KEY,
    video_id CHARACTER(11) REFERENCES videos (id),
    label_id CHARACTER(20) REFERENCES labels (id),
    UNIQUE (video_id, label_id)
);

CREATE TABLE filters (
    id serial PRIMARY KEY,
    comments CHARACTER(200)
);

CREATE TABLE embeddings (
    id serial PRIMARY KEY,
    video_id CHARACTER(11) NOT NULL REFERENCES videos (id),
    filter_id integer NOT NULL REFERENCES filters (id),
    worker CHARACTER(200),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    aws_key CHARACTER(60),
    UNIQUE (video_id, filter_id),
    UNIQUE (aws_key)
);
"""

def main(conn):
    cur = conn.cursor()
    hostname = socket.gethostname()
    my_limit = 1
    sql = """
    UPDATE embeddings SET worker=%s, start_time=now() FROM
        (SELECT id, video_id, filter_id from embeddings
        WHERE start_time IS NULL LIMIT %s) sub
    INNER JOIN videos on videos.id=sub.video_id
    INNER JOIN filters on filters.id=sub.filter_id
    WHERE embeddings.id=sub.id AND filters.id=sub.filter_id
    RETURNING sub.id, sub.video_id, sub.filter_id, videos.start_time, videos.end_time, filters.coefficients;
    """
    cur.execute(sql, (hostname, my_limit))
    conn.commit()
    rows = cur.fetchall()
    tf.Graph().as_default()
    sess = tf.Session()

    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.

    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, CHECKPOINT)

    tmpdir = tempfile.TemporaryDirectory()
    for row in rows:
        id, video_id, filter_id, start_time, end_time, b_n = row
        print(video_id)
        tmpfile = yt_dl(video_id, start_time, end_time, 44100, tmpdir.name, b_n)
        if tmpfile is None:
            comment = "Youtube download failed"
            sql = 'UPDATE embeddings set end_time=now(), comment=%s WHERE id=%s'
            cur.execute(sql, (comment, id))
            conn.commit()
            continue
        arr = wav2vggish(tmpfile, sess)
        tmpfile = tempfile.NamedTemporaryFile(suffix='.npy')
        np.save(tmpfile, arr)
        aws_key = f'filter{filter_id}/{video_id}.npy'
        upload_to_aws(tmpfile.name, aws_key)
        tmpfile.close()
        sql = 'UPDATE embeddings set end_time=now(), aws_key=%s WHERE id=%s'
        cur.execute(sql, (aws_key, id))
        conn.commit()

    sess.close()

def yt_dl(yt_id, t_start, t_end, Fs, dir, b_n):
    ydl_opts = {
        'outtmpl': f'{dir}/{yt_id}.wav',
        'format': 'bestaudio/best',
        'logger': logging,
        'postprocessors': [{
             'key': 'FFmpegExtractAudio',
             'preferredcodec': 'wav',
             'preferredquality': '192',
        }]
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['https://www.youtube.com/watch?v=%s' % yt_id])

            # Read .wav
            y, sr = lib.load(f'{dir}/{yt_id}.wav')
            # Crop .wav
            y = y[int(t_start*sr):int(t_end*sr)]
            # Resample .wav
            y_resampled = lib.resample(y, sr, Fs)
            y_resampled = lib.util.fix_length(y_resampled, int((10)*Fs))
            print(f'y_resampled length = {y_resampled.shape}')
            # filter .wav
            y_filtered = np.convolve(y_resampled, b_n, 'same')

            # Store original and filtered file
            clipped_file = f'{dir}/{yt_id}_clipped.wav'
            filtered_file = f'{dir}/{yt_id}_filtered.wav'
            lib.output.write_wav(clipped_file, y_resampled, Fs)
            lib.output.write_wav(filtered_file, y_filtered, Fs)

    except youtube_dl.utils.DownloadError:
        logging.error(f'ID: {yt_id} failed to download')
        return None
    return filtered_file

def upload_to_aws(file, key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

    bucket_name = 'cs230-deep-audio'
    local_filename = str(file)
    remote_filename = key
    s3.upload_file(local_filename, bucket_name, remote_filename)

def wav2vggish(file, sess):
    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    examples_batch = w2e(file)
    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    postprocessed_batch = pproc.postprocess(embedding_batch)

    # Numpy array of shape (`t`, 128)
    return postprocessed_batch



def populate(conn):

    cur = conn.cursor()

    url = 'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json'
    ontology = requests.get(url).content.decode()
    ontology = json.loads(ontology)
    for label in ontology:
        break # We already did this once....
        id = label['id']
        display_name = label['name']
        sql = 'INSERT INTO labels (id, display_name) VALUES (%s, %s);'
        cur.execute(sql, (label['id'], label['name']))
        conn.commit()
        print(id)


    # BALANCED TRAIN
    urls = [
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv'
        ]
    for url in urls:
        r = requests.get(url)
        lines = r.content.decode()
        lines = lines.split('\n')[3:]
        for line in lines:
            try:
                line, labels, _ = line.split('"')
                slug, start, end = line.split(',')[0:3]
                labels = labels.split(',')
                start = float(start)
                end = float(end)
                sql = 'INSERT INTO videos (id, start_time, end_time) VALUES (%s, %s, %s) returning id;'
                cur.execute(sql, (slug, start, end))
                video_id = cur.fetchone()
                print(video_id)
                for label in labels:
                    sql = 'INSERT INTO labels_videos (video_id, label_id) VALUES (%s, %s);'
                    cur.execute(sql, (video_id, label))
                conn.commit()
                #print(line, start, end, labels)
            except ValueError as e:
                print('error' + line)
                print(e)

if __name__ == '__main__':
    with psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD) as conn:

        # Only do this once from a main computer
        #populate(conn)
        main(conn)
