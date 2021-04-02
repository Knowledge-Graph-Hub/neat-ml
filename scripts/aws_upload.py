from neat.upload import upload

upload.upload_dir_to_s3(local_directory=local_dir, s3_bucket=my_s3_bucket, s3_bucket_dir=my_remote_dir")
