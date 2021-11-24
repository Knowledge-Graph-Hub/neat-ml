from unittest import TestCase

from neat.pre_run_checks.pre_run_checks import pre_run_checks
from neat.yaml_helper.yaml_helper import YamlHelper
from unittest import mock


class TestPreRunChecks(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.upload_yaml = YamlHelper('tests/resources/test_upload_full.yaml')

    @mock.patch("boto3.client")
    def test_pre_run_check_good_credentials(self, mock_boto_client) -> None:
        return_val = pre_run_checks(self.upload_yaml, check_s3_credentials=True)
        self.assertTrue(return_val)
        self.assertTrue(mock_boto_client.called)







