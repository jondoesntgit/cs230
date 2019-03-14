#!/usr/bin/env python

"""
Fetches a worklist from a databse and converts youtube slugs to vggish features
and stores them on AWS
"""
import sys
sys.path.append('..')
from pathlib import Path
from dotenv import load_dotenv
import os
import boto3
import librosa as lib

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

def main(conn, num_videos_to_grab):
    cur = conn.cursor()
    hostname = socket.gethostname()
    sql = """
    SELECT sub.id, sub.video_id, sub.filter_id, videos.start_time, videos.end_time, filters.coefficients FROM
        (SELECT id, video_id, filter_id from embeddings
        WHERE aws_wav_key IS NULL LIMIT %s) sub
    INNER JOIN videos on videos.id=sub.video_id
    INNER JOIN filters on filters.id=sub.filter_id
    -- WHERE embeddings.id=sub.id AND filters.id=sub.filter_id
    ;
    """
    cur.execute(sql, (num_videos_to_grab,))
    conn.commit()
    rows = cur.fetchall()
    logging.info(f'Fetched {len(rows)} videos')

    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.

    tmpdir = tempfile.TemporaryDirectory()
    for row in rows:
        id, video_id, filter_id, start_time, end_time, b_n = row
        logging.info(f'Processing {video_id}')
        tmpdir_name = tmpdir.name
        clipped_file, filtered_file = yt_dl(video_id, start_time, end_time, 44100, tmpdir_name, b_n)
        if filtered_file is None:
            comment = "Youtube download failed: %s" % video_id
            sql = 'INSERT INTO errors (timestamp, description) VALUES(timestamp=now(), description=%s);s'
            cur.execute(sql, (comment))
            conn.commit()
            continue
        aws_wav_key = f'wav/f{filter_id}/{video_id}.wav'
        logging.info(f'Uploading from {filtered_file}')
        upload_to_aws(filtered_file, aws_wav_key)
        upload_to_aws(clipped_file, f'wav/fraw/{video_id}.wav')
        logging.info('Upload complete')
        sql = 'UPDATE embeddings SET aws_wav_key=%s WHERE id=%s'
        cur.execute(sql, (aws_wav_key, id))
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
    return clipped_file, filtered_file

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

def populate(conn):
    print('Populating database')
    cur = conn.cursor()

    url = 'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json'
    ontology = requests.get(url).content.decode()
    ontology = json.loads(ontology)
    print('Populating labels')
    for label in ontology:
        break # We already did this once....
        id = label['id']
        display_name = label['name']
        sql = 'INSERT INTO labels (id, display_name) VALUES (%s, %s) ON CONFLICT DO NOTHING;'
        cur.execute(sql, (label['id'], label['name']))
        conn.commit()
        print(id)

    url = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
    class_labels_indices = requests.get(url).content.decode()
    for line in class_labels_indices.split('\n'):
        try:
            splits = line.split(',')
            id_number = int(splits[0])
            id_string = splits[1]
            sql = 'UPDATE labels SET index=%s WHERE id=%s;'
            cur.execute(sql, (id_number, id_string))
            conn.commit()
            print(id_number, id_string)
        except:
            continue

    return

    print('Populating videos')
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
                sql = 'INSERT INTO videos (id, start_time, end_time) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING returning id;'
                cur.execute(sql, (slug, start, end))
                video_id = cur.fetchone()
                print(video_id)
                if video_id is None:
                    continue
                for label in labels:
                    sql = 'INSERT INTO labels_videos (video_id, label_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;'
                    cur.execute(sql, (video_id, label))
                conn.commit()
                #print(line, start, end, labels)
            except ValueError as e:
                print('error' + line)
                print(e)

if __name__ == '__main__':
    logging.info('Starting')
    with psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD) as conn:

        # Only do this once from a main computer
        #populate(conn)
        #exit()
        #try:
        main(conn, 20)
        #except Exception as e:
        #    logging.error(e)
        #    cur = conn.cursor()
        #    cur.execute('INSERT INTO errors (description) VALUES (%s);', (str(e),))
        #    conn.commit()

