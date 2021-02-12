from unittest import TestCase
import boto3
from botocore.stub import Stubber
from neat.upload.upload import upload_dir_to_s3
from neat.yaml_helper.yaml_helper import YamlHelper


class TestUpload(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.boto3 = boto3.client('s3')

    def setUp(self) -> None:
        self.bad_yaml = 'tests/resources/test_bad_upload_info.yaml'
        self.good_yaml = 'tests/resources/test_good_upload_info.yaml'
        self.stubber = Stubber(self.boto3)
        # self.stubber.add_response('', self.upload_file_response,
        #                           self.expected_upload_file_params)

    def test_boto_list_buckets(self) -> None:
        with self.stubber:
            expected_ls_params = {}
            list_buckets_response = {
                "Owner": {"DisplayName": "name", "ID": "EXAMPLE123"},
                "Buckets":
                    [{"CreationDate": "2016-05-25T16:55:48.000Z", "Name": "foo"}]}
            self.stubber.add_response('list_buckets', list_buckets_response,
                                      expected_ls_params)
            self.assertEqual(self.boto3.list_buckets(), list_buckets_response)

    # def test_upload_dir_to_s3(self) -> None:
    #     with self.stubber:
    #         self.assertEqual(self.boto3.list_buckets(), self.list_buckets_response)
    #
    #         upload_dir_to_s3(local_directory, bucket, destination)
