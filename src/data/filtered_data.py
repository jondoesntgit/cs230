#!/usr/bin/python


import numpy as np
import boto3
import psycopg2
from optparse import OptionParser
import tqdm
import dotenv
import os
from pathlib import Path
import tempfile

dotenv.load_dotenv()

PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
PSQL_USERNAME = os.getenv('PSQL_USERNAME')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
PSQL_DATABASE = os.getenv('PSQL_DATABASE')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = 'cs230-deep-audio'

def get_aws_keys(filter_id):
    conn = psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD)

    cur = conn.cursor()
    query = '''
    SELECT video_id, aws_key FROM embeddings 
    WHERE aws_key IS NOT NULL 
    AND filter_id = %s;
    '''
    cur.execute(query, (filter_id,));

    return_dict = {
        row[0]: row[1].strip() for row in cur.fetchall()
    }

    conn.close()
    return return_dict

def download_embeddings(aws_keys_dict, local_dir=None, verbose=True):
    if local_dir is None:
        local_dir = tempfile.TemporaryDirectory()
    else:
        Path(local_dir).mkdir(parents=True, exists_ok=True)

    iter_list = aws_keys.values()
    if verbose:
        iter_list = tqdm.tqdm(iter_list)
    for remote_filename in iter_list:
        bucket_name = 
        s3.download_file(BUCKET_NAME, remote_filename, local_filename)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--filter', dest='filter_id', type="int", help='The id of the filter to query')
    parser.add_option('-q', '--quiet', dest='quiet', help='Suppress debug information')
    parser.add_option('-o', '--outfile', dest='output_file', type="string", help='File to output all the concatenated data to')

    options, args = parser.parse_args()

    #print("options", options)
    #print("args", args)

    filter_id = options.filter_id

    if filter_id is None:
        raise ValueError('Filter id must be set using somethign like --filter=1')

    aws_key_dict = get_aws_keys(filter_id=filter_id)
    print(aws_key_dict)

