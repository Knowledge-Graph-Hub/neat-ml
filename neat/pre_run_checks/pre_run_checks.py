import warnings

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from neat.yaml_helper.yaml_helper import YamlHelper


def pre_run_checks(yhelp: YamlHelper,
                   check_s3_credentials: bool = True,
                   check_s3_bucket: bool = True,
                   check_s3_bucket_dir: bool = True,
                   ) -> bool:
    """Some checks before run, to prevent frustrating failure at the end of long runs

    Args:
        yhelp: YamlHelper object
        check_s3_credentials: should we check S3 credentials (true). Note that if no
            upload dir exists, this will pass
        check_s3_bucket: check that s3 bucket is valid
        check_s3_bucket_dir: check that s3 bucket directory doesn't already exist

    Returns:
        Boolean pass or fail
    """
    return_val = True
    if check_s3_credentials:
        try:
            client = boto3.client('s3')
            client.list_buckets()  # to check credentials
        except ClientError as ce:
            warnings.warn(f"Client error when trying S3 credentials: {ce}")
            if yhelp.do_upload():
                return_val = False
            else:
                warnings.warn("YAML contains no upload block - continuing")

    if check_s3_bucket:
        pass
    if check_s3_bucket_dir:
        pass

    return return_val
