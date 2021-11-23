import logging
import os
import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore


def upload_dir_to_s3(local_directory: str, s3_bucket: str, s3_bucket_dir: str,
                     extra_args=None) -> None:

    client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):

        for filename in files:
            local_path = os.path.join(root, filename)

            # construct the full path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_bucket_dir, relative_path)

            logging.info(f"Searching {s3_path} in {s3_bucket}")
            try:
                client.head_object(Bucket=s3_bucket, Key=s3_path)
                logging.warning("Path found on S3! Skipping {s3_path}")
            except ClientError:  # Exception abuse
                logging.info(f"Uploading {s3_path}")
                client.upload_file(local_path, s3_bucket, s3_path,
                                   ExtraArgs=extra_args)
