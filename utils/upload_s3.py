import math
import os
import sys
from boto.s3.connection import S3Connection
from filechunkio import FileChunkIO

# Constanst variables
KEY = 'AKIAJUCKLBAF2TYNZ7IQ'
PWD = 'aZjhbof3dSbOzEG6qEzxn/ZWruoTSrSpCOsdgV+N'
BUCKET = 'oliverdelacruz'
PATHS = ['/home/ubuntu/minecraft/logs', '/home/ubuntu/scalable_agent/logs']
# PATHS = ['/home/ubuntu/minecraft/logs', '/home/ubuntu/scalable_agent/logs' ]
# PATHS = ['/home/deoliver/minecraft/logs', '/home/deoliver/scalable_agent/logs' ]

# Check available buckets
conn = S3Connection(KEY, PWD)
rs = [b.name for b in conn.get_all_buckets()]

def insert(path_folder, bucket):
    for path, subdirs, files in os.walk(path_folder):
        for name in files:
            source_path = os.path.join(path, name)
            dest_path = os.path.join('/'.join(path.split('/')[-2:]), name)
            try:
                # Get file info
                source_size = os.stat(source_path).st_size

                # Create a multipart upload request
                mp = bucket.initiate_multipart_upload(dest_path)

                # Use a chunk size of 50 MiB (feel free to change this)
                chunk_size = 52428800
                chunk_count = int(math.ceil(source_size / float(chunk_size)))

                # Send the file parts, using FileChunkIO to create a file-like object
                # that points to a certain byte range within the original file. We
                # set bytes to never exceed the original file size.
                for i in range(chunk_count):
                    offset = chunk_size * i
                    bytes = min(chunk_size, source_size - offset)
                    with FileChunkIO(source_path, 'r', offset=offset, bytes=bytes) as fp:
                        mp.upload_part_from_file(fp, part_num=i + 1)

                # Finish the upload
                mp.complete_upload()
            except Exception as e:
                print(e)
                pass

if BUCKET in rs:
    b = conn.get_bucket(BUCKET)
    [insert(path, b) for path in PATHS]
else:
    sys.exit()