from unittest import TestCase
from botocore.exceptions import ClientError
from neat.upload.upload import upload_dir_to_s3
from neat.yaml_helper.yaml_helper import YamlHelper
from unittest import mock


class TestUpload(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.bad_yaml = 'tests/resources/test_bad_upload_info.yaml'
        self.good_yaml = 'tests/resources/test_good_upload_info.yaml'

    @mock.patch("boto3.client.head_object")
    @mock.patch("boto3.client.upload_file")
    @mock.patch("boto3.client")
    def test_upload_dir_to_s3(self, mock_boto_client, mock_upload_file, mock_head_object):
        mock_upload_file.return_value = {}
        mock_head_object.return_value = ClientError({'Error':
                                                     {'Code': '42',
                                                      'Message': "don't kill bugs"}},
                                                   "head_object")
        mock_upload_file.upload_file = {}
        kwargs = YamlHelper(self.good_yaml).make_upload_args()
        upload_dir_to_s3(**kwargs)
        self.assertEqual(mock_boto_client.call_count, 1)

