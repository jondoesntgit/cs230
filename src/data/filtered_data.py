#!/usr/bin/env python


import numpy as np
import boto3
import psycopg2
from optparse import OptionParser
import tqdm
import dotenv
import os
from pathlib import Path
import tempfile
import h5py

dotenv.load_dotenv()

PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
PSQL_USERNAME = os.getenv('PSQL_USERNAME')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
PSQL_DATABASE = os.getenv('PSQL_DATABASE')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = 'cs230-deep-audio'

def get_aws_keys(filter_id, limit=None):
    conn = psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD)

    cur = conn.cursor()
    query = '''
    SELECT video_id, aws_key, array(
        SELECT labels.index FROM labels_videos
        INNER JOIN labels ON labels_videos.label_id = labels.id
        WHERE embeddings.video_id=labels_videos.video_id
    ) FROM embeddings
    WHERE aws_key IS NOT NULL
    AND filter_id = %s
    LIMIT %s;
    '''
    cur.execute(query, (filter_id, limit));

    return_dict = {
        row[0]: {"aws_key": row[1].strip(), "labels": row[2]} for row in cur.fetchall()
    }

    conn.close()
    return return_dict


def download_embeddings(aws_keys_dict, local_dir=None, verbose=True):
    """Return a list of downloaded filenames."""
    if local_dir is None:
        local_dir = tempfile.TemporaryDirectory().name
        if verbose:
            print('Writing to %s' % local_dir)
    else:
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        's3',
        # Hard coded strings as credentials, not recommended.
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    iter_list = [aws_keys_dict[key]['aws_key'] for key in aws_keys_dict]
    labels_list = [aws_keys_dict[key]['labels'] for key in aws_keys_dict]
    # Convert to multi-hot
    multi_hot_labels = np.zeros((len(labels_list), 527))
    for i, labels in enumerate(labels_list):
        for l in labels:
            multi_hot_labels[i,l] = 1

    ret_list = []
    if verbose:
        iter_list = tqdm.tqdm(iter_list)
    for remote_filename in iter_list:
        local_filename = Path(local_dir) / remote_filename
        local_filename.parent.mkdir(parents=True, exist_ok=True)
        ret_list.append(local_filename)
        s3.download_file(BUCKET_NAME, remote_filename, str(local_filename))
    return ret_list, multi_hot_labels


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--filter', dest='filter_id', type="int", help='The id of the filter to query')
    parser.add_option('-q', '--quiet', dest='quiet', help='Suppress debug information')
    parser.add_option('-o', '--outfile', dest='output_file', type="string", help='File to output all the concatenated data to (h5)')
    parser.add_option('-l', '--limit', dest='limit', type='int', help='Limit the total number of examples. Defaults to all examples')
    parser.add_option('-d', '--localdir', dest='local_directory', type='string', default=None, help='A working directory to write all of the embeddings to. Defaults to a temporary directory.')

    options, args = parser.parse_args()

    #print("options", options)
    #print("args", args)

    filter_id = options.filter_id
    output_file = options.output_file
    output_file = str(Path(output_file).expanduser())
    limit = options.limit
    local_dir = options.local_directory

    if filter_id is None:
        raise ValueError('Filter id must be set using somethign like --filter=1')

    if output_file is None:
        raise ValueError('Output file must be set to some .h5 file using something like --outfile=~/data.h5')

    output_file_as_path = Path(output_file)
    output_file_as_path.parent.mkdir(parents=True, exist_ok=True)

    aws_key_dict = get_aws_keys(filter_id=filter_id, limit=limit)
    npy_files, multi_hot_labels = download_embeddings(
        aws_key_dict, local_dir=local_dir)

    X = np.stack([np.load(npy_file) for npy_file in npy_files])
    y = multi_hot_labels
    slugs = np.array(list(aws_key_dict.keys())).astype('S11')

    h5file = h5py.File(str(output_file), mode='w')
    h5file['X'] = X
    h5file['y'] = y
    h5file['slugs'] = slugs
    h5file.close()

