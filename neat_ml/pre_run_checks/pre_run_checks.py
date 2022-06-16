import collections
import warnings

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from neat_ml.yaml_helper.yaml_helper import YamlHelper

def pre_run_checks(
    yhelp: YamlHelper,
    check_s3_credentials: bool = True,
    check_s3_bucket: bool = True,
    check_s3_bucket_dir: bool = True,
    check_classifiers: bool = True,
) -> bool:
    """Some checks before run, to prevent frustrating failure at the end of long runs

    Args:
        yhelp: YamlHelper object
        check_s3_credentials: should we check S3 credentials (true). Note that if no
            upload dir exists, this will pass
        check_s3_bucket: check that s3 bucket exists on s3
        check_s3_bucket_dir: check that s3 bucket directory doesn't already exist
        check_classifiers: verify that classifier ids don't conflict

    Returns:
        Boolean pass or fail
    """
    return_val = True

    if check_s3_credentials:
        try:
            client = boto3.client("s3")
            client.list_buckets()  # to check credentials
        except ClientError as ce:
            warnings.warn(f"Client error when trying S3 credentials: {ce}")
            if yhelp.do_upload():
                return_val = False
            else:
                warnings.warn("YAML contains no upload block - continuing")

    if (
        check_s3_bucket and yhelp.do_upload()
    ):  # make sure we are going to upload
        upload_args = yhelp.make_upload_args()
        try:
            client = boto3.client("s3")
            buckets = []
            bucket_info = client.list_buckets()
            if "Buckets" in bucket_info:
                buckets = [
                    this_dict["Name"] for this_dict in bucket_info["Buckets"]
                ]
            else:
                warnings.warn(
                    "Can't find 'Buckets' key in output of client.list_buckets()"
                )
            if "s3_bucket" not in upload_args:
                warnings.warn("No 's3_bucket' in upload block")
                return_val = False
            elif upload_args["s3_bucket"] not in buckets:
                warnings.warn(
                    f"Can't find the target s3 bucket "
                    f"{upload_args['s3_bucket']} in s3"
                )
                return_val = False
        except ClientError as ce:
            warnings.warn(f"Client error when trying S3 credentials: {ce}")
            return_val = False

    if (
        check_s3_bucket_dir and yhelp.do_upload()
    ):  # make sure we are going to upload
        upload_args = yhelp.make_upload_args()
        
        if not pre_bucket_check(upload_args):
            return_val = False

    if check_classifiers and yhelp.classifiers():
        # PSEUDOCODE
        # if classifiers:
        # Then make sure id-s aren't clashing
        # &
        # classifier_model_id is calling a valid id.

        all_classifier_ids = yhelp.get_all_classifier_ids()
        if len(all_classifier_ids) != len(set(all_classifier_ids)):
            dup_ids = [
                item # type: ignore
                for item, count in collections.Counter(  # type: ignore
                    all_classifier_ids
                ).items()
                if count > 1
            ]

            raise ValueError(
                f"Same 'classifier_id' represents multiple classes in the yaml provided: {dup_ids}"
            )

        if yhelp.do_apply_classifier():
            check = all(
                item in all_classifier_ids
                for item in yhelp.get_classifier_id_for_prediction()
            )
            if not check:
                return_val = False
                raise ValueError(
                    f"The 'classifier_id' used for prediction does "
                    "not map to any classifier in the yaml provided:"
                    "{yhelp.get_classifier_id_for_prediction()}"
                )

    return return_val

def pre_bucket_check(upload_args: dict) -> bool:
    """
    Given upload args, checks if the
    target bucket and directory are
    accessible and empty, respectively.
    """

    success = True

    try:
        client = boto3.client("s3")
        if "s3_bucket_dir" not in upload_args:
            warnings.warn("No 's3_bucket_dir' in upload block")
            success = False
        else:
            result = client.list_objects(
                Bucket=upload_args["s3_bucket"],
                Prefix=upload_args["s3_bucket_dir"],
            )
            if "Contents" in result:
                warnings.warn(
                    f"There are already objects in remote s3 directory: "
                    f"{upload_args['s3_bucket_dir']}"
                )
                success = False
    except ClientError as ce:
        warnings.warn(f"Client error when trying S3 credentials: {ce}")
        success = False

    return success

if __name__ == "__main__":
    pre_run_checks(yhelp=YamlHelper())  # type: ignore
