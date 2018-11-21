import math
import os
import sys
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
import re

# Constanst variables
KEY = 'username'
PWD = 'password'
BUCKET = 'bucket'
PATHS = ['/home/ubuntu/']


#PATHS = ['C:\Thesis\scalable_agent\logs']
# Check available buckets
conn = S3Connection(KEY, PWD)
rs = [b.name for b in conn.get_all_buckets()]
KEYS = ['ip', 'csv', 'txt', 'checkpoint', 'graph']

def download(path_folder, bucket):
    bucket_list = bucket.list()
    for l in bucket_list:
        key_string = str(l.key)
        s3_path = path_folder + '/' + key_string
        folder_s3 = '/'.join(s3_path.split('/')[:-1])
        file_string = re.split('.|/|-', key_string)

        if any(key_word in key_string for key_word in KEY) and not ('model' in key_string):
            # check if the folder exists
            if not os.path.exists(folder_s3):
                os.makedirs(folder_s3)

            # check if the file exists
            if not os.path.exists(s3_path):
                # download file
                try:
                    l.get_contents_to_filename(s3_path)
                except (OSError, S3ResponseError) as e:
                    pass

if BUCKET in rs:
    b = conn.get_bucket(BUCKET)
    [download(path, b) for path in PATHS]
else:
    sys.exit()
