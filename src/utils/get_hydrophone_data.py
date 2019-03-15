#!/usr/bin/env python

import sys
import os
import boto3
import psycopg2
import socket
import dotenv
import subprocess
dotenv.load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
PSQL_USERNAME = os.getenv('PSQL_USERNAME')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
PSQL_DATABASE = os.getenv('PSQL_DATABASE')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def get_a_video_id(conn):
    hostname = socket.gethostname()
    cur = conn.cursor()
    cur.execute('''
        UPDATE embeddings SET start_time=now(), worker=%s
        WHERE id IN (
            SELECT id FROM embeddings WHERE
            filter_id=3 AND aws_wav_key IS NULL
            LIMIT 1)
        RETURNING video_id;
        ''', (hostname, ))
    rows = cur.fetchall()
    video_id = rows[0][0]
    conn.commit()
    return video_id

def download_raw_wav(video_id, local_filename, s3):
    bucket_name = 'cs230-deep-audio'
    remote_filename = 'wav/fraw/%s.wav' % video_id
    s3.download_file(bucket_name, remote_filename, local_filename)

def play_video_and_record(raw_filename, wav_filename, numpy_filename):
    subprocess.call(['python', 'play_through_hydrophone.py', 
        '-f', raw_filename, 
        '-t', wav_filename, 
        '-n', numpy_filename])

def upload_recording(video_id, wav_filename, numpy_filename, s3):
    bucket_name = 'cs230-deep-audio'
    remote_filename = 'wav/f3/%s.wav' % video_id
    s3.upload_file(wav_filename, bucket_name, remote_filename)

    remote_filename = 'npy/f3/%s.npy' % video_id
    s3.upload_file(numpy_filename, bucket_name, remote_filename)


def update_database(conn, video_id):
    cur = conn.cursor()
    aws_wav_key = 'wav/f3/%s.wav' % video_id
    cur.execute('''
        UPDATE embeddings SET end_time=now(), aws_wav_key=%s
        WHERE filter_id=3 AND video_id=%s
        ''', (aws_wav_key, video_id))
    conn.commit()
    return

def process_a_video(conn, s3):
    video_id = get_a_video_id(conn)
    logging.info('Processing %s' % video_id)
    local_filename = 'working_video.wav'
    wav_filename = 'hydrophone.wav'
    numpy_filename = 'numpy.npy'
    download_raw_wav(video_id, local_filename, s3)
    
    raw_filename = local_filename
    play_video_and_record(
        raw_filename=raw_filename, 
        wav_filename=wav_filename, 
        numpy_filename=numpy_filename)

    upload_recording(video_id, wav_filename, numpy_filename, s3)
    update_database(conn, video_id)

def main():

    conn = psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD)
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    while True:
        process_a_video(conn, s3)
    conn.close()

if __name__ == '__main__':
    main()