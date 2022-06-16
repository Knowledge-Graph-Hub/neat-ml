from unittest import TestCase
from unittest.mock import MagicMock, patch

import boto3
from botocore.exceptions import ClientError
from neat_ml.upload.upload import upload_dir_to_s3
from neat_ml.yaml_helper.yaml_helper import YamlHelper
from unittest import mock


class TestUpload(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.bad_yaml = 'tests/resources/test_bad_upload_info.yaml'
        self.good_yaml = 'tests/resources/test.yaml'
        self.good_kwargs = YamlHelper(self.good_yaml).make_upload_args()

    @mock.patch("boto3.client")
    def test_upload_dir_to_s3_calls_boto3(self, mock_boto_client):
        upload_dir_to_s3(**self.good_kwargs)
        self.assertTrue(mock_boto_client.called)

    @mock.patch("boto3.client")
    def test_upload_dir_to_s3_starts_s3_client(self, mock_boto_client):
        upload_dir_to_s3(**self.good_kwargs)
        self.assertTrue(mock_boto_client.called)
        self.assertEqual(mock_boto_client.call_args[0][0], 's3')





