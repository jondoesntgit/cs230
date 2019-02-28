#!/usr/bin/env python
import boto3
from dotenv import load_dotenv
import os
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client(
    's3',
    # Hard coded strings as credentials, not recommended.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
local_filename = 'file.txt'
remote_filename = 'file.txt'
bucket_name = 'cs230-deep-audio'

s3.upload_file(local_filename, bucket_name, remote_filename)

local_filename = '/tmp/foo.txt'
remote_filename = 'fixtures/file2.txt'
s3.download_file(bucket_name, remote_filename, local_filename)

with open(local_filename, 'r') as f:
    print(f.read())