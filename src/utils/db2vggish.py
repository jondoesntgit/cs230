#!/usr/bin/env python

"""
Fetches a worklist from a databse and converts youtube slugs to vggish features
and stores them on AWS
"""

import psycopg2
import os
import dotenv
import requests
import json
dotenv.load_dotenv()

PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
PSQL_USERNAME = os.getenv('PSQL_USERNAME')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
PSQL_DATABASE = os.getenv('PSQL_DATABASE')

CREATE_TABLES = """

CREATE TYPE split AS ENUM ('train', 'dev', 'test');

CREATE TABLE videos (
    id serial PRIMARY KEY,
    slug CHARACTER(11) UNIQUE,
    start_time integer,
    end_time integer,
    split split,
    UNIQUE (slug, start_time, end_time)
);

CREATE TABLE labels (
    id CHARACTER(20) PRIMARY KEY,
    display_name CHARACTER(50) UNIQUE,
    parent_id CHARACTER(20) REFERENCES labels (id)
);

CREATE TABLE labels_videos (
    id serial PRIMARY KEY,
    video_id integer REFERENCES videos (id),
    label_id CHARACTER(20) REFERENCES labels (id),
    UNIQUE (video_id, label_id)
);

CREATE TABLE filters (
    id serial PRIMARY KEY,
    comments CHARACTER(200)
);

CREATE TABLE jobs (
    id serial PRIMARY KEY,
    video_id integer NOT NULL REFERENCES videos (id),
    filter_id integer NOT NULL REFERENCES filters (id),
    worker CHARACTER(200),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    UNIQUE (video_id, filter_id)
);

CREATE TABLE embeddings (
    id serial PRIMARY KEY,
    aws_key character(60) NOT NULL ,
    video_id integer NOT NULL REFERENCES videos (id),
    filter_id integer NOT NULL REFERENCES filters (id),
    job_id integer NOT NULL REFERENCES jobs (id),
    UNIQUE (video_id, filter_id)
);
"""

def main(conn):
    print(conn)

def populate(conn):

    cur = conn.cursor()

    url = 'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json'
    ontology = requests.get(url).content.decode()
    ontology = json.loads(ontology)
    """
    for label in ontology:
        break # We already did this once....
        id = label['id']
        display_name = label['name']
        sql = 'INSERT INTO labels (id, display_name) VALUES (%s, %s);'
        cur.execute(sql, (label['id'], label['name']))
        conn.commit()
        print(id)
        """


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
                sql = 'INSERT INTO videos (slug, start_time, end_time) VALUES (%s, %s, %s) returning id;'
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

        populate(conn)
        main(conn)
