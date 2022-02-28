from unittest import TestCase
from botocore.exceptions import ClientError

from neat.pre_run_checks.pre_run_checks import pre_run_checks
from neat.yaml_helper.yaml_helper import YamlHelper
from unittest import mock


class TestPreRunChecks(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.upload_yaml = YamlHelper("tests/resources/test_upload_full.yaml")
        self.classifier_yaml = YamlHelper(
            "tests/resources/test_classifiers.yaml"
        )

    @mock.patch("boto3.client")
    def test_pre_run_check_confirm_boto_called(self, mock_boto_client) -> None:
        return_val = pre_run_checks(
            self.upload_yaml,
            check_s3_credentials=True,
            check_classifiers=False,
        )
        self.assertTrue(mock_boto_client.called)

    @mock.patch("boto3.client")
    def test_pre_run_check_bad_credentials(self, mock_boto_client) -> None:
        mock_boto_client.side_effect = ClientError(
            error_response=mock_boto_client, operation_name=mock_boto_client
        )
        # returns false if we have an upload key in yaml
        return_val = pre_run_checks(
            self.upload_yaml,
            check_s3_credentials=True,
            check_classifiers=False,
        )
        self.assertFalse(return_val)
        self.assertTrue(mock_boto_client.called)

    @mock.patch("boto3.client")
    def test_pre_run_check_bad_credentials_but_no_upload(
        self, mock_boto_client
    ) -> None:
        mock_boto_client.side_effect = ClientError(
            error_response=mock_boto_client, operation_name=mock_boto_client
        )
        return_val = pre_run_checks(
            YamlHelper("tests/resources/test_no_upload.yaml"),
            check_s3_credentials=True,
            check_classifiers=False,
        )
        # returns true if bad creds, but we don't have upload key in yaml
        self.assertTrue(return_val)
        self.assertTrue(mock_boto_client.called)

    @mock.patch("boto3.client")
    def test_pre_run_check_bad_credentials_but_no_check(
        self, mock_boto_client
    ) -> None:
        mock_boto_client.side_effect = ClientError(
            error_response=mock_boto_client, operation_name=mock_boto_client
        )
        return_val = pre_run_checks(
            YamlHelper("tests/resources/test_no_upload.yaml"),
            check_s3_credentials=False,
            check_classifiers=False,
        )
        # returns true if bad creds, but we don't want to check credentials
        self.assertTrue(return_val)

    def test_pre_run_classifiers(self) -> None:
        return_val = pre_run_checks(
            self.classifier_yaml,
            check_s3_credentials=False,
            check_s3_bucket=False,
            check_s3_bucket_dir=False,
            check_classifiers=True,
        )

    # TODO:
    # 1. Test wrong classifier_id
    # 2. test duplicate ids for classifiers
    # 3. Check mandatory properties exist.
