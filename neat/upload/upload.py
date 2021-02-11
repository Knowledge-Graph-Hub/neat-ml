import logging
import os
import boto3
from botocore.exceptions import ClientError


def upload_dir_to_s3(local_directory: str, bucket: str, destination: str) -> None:

    client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):

        for filename in files:
            local_path = os.path.join(root, filename)

            # construct the full path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)

            logging.info(f"Searching {s3_path} in {bucket}")
            try:
                client.head_object(Bucket=bucket, Key=s3_path)
                logging.warning("Path found on S3! Skipping {s3_path}")
            except ClientError:  # Exception abuse
                logging.info(f"Uploading {s3_path}")
                client.upload_file(local_path, bucket, s3_path)
