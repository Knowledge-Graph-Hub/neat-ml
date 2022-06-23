import re
from unittest import TestCase, mock

from click.testing import CliRunner

from neat_ml.cli import run, updateyaml


class TestRun(TestCase):
    """Tests the neat.py script."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_run_no_yaml_file(self):
        result = self.runner.invoke(
            catch_exceptions=False, cli=run, args=["--config", "doesntexist"]
        )
        self.assertTrue(re.search("doesntexist", result.output))
        self.assertNotEqual(result.exit_code, 0)

    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("boto3.client")
    @mock.patch("neat_ml.cli.pre_run_checks")
    def test_run_do_upload(
        self,
        mock_pre_run_checks,
        mock_boto3_client,
        mock_do_embeddings,
        mock_do_tsne,
        mock_do_classifier,
        mock_do_upload,
    ):
        mock_do_embeddings.return_value = False
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = True
        mock_pre_run_checks.return_value = True
        result = self.runner.invoke(
            catch_exceptions=False,
            cli=run,
            args=["--config", "tests/resources/test.yaml"],
        )
        self.assertTrue(mock_do_upload.called)
        self.assertTrue(mock_boto3_client.called)
        self.assertEqual(result.exit_code, 0)

    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("neat_ml.graph_embedding.graph_embedding.make_node_embeddings")
    @mock.patch("boto3.client")
    def test_run_do_embeddings(
        self,
        mock_boto,
        mock_make_node_embeddings,
        mock_do_embeddings,
        mock_do_tsne,
        mock_do_classifier,
        mock_do_upload,
    ):
        mock_do_embeddings.return_value = True
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = False
        result = self.runner.invoke(
            catch_exceptions=False,
            cli=run,
            args=["--config", "tests/resources/test.yaml"],
        )
        self.assertEqual(result.exit_code, 0)

    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_upload")
    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    # @mock.patch("neat_ml.visualization.visualization.make_tsne")
    # @mock.patch("boto3.client")
    # def test_run_do_tsne(
    #     self,
    #     mock_boto,
    #     mock_make_tnse,
    #     mock_do_embeddings,
    #     mock_do_tsne,
    #     mock_do_classifier,
    #     mock_do_upload,
    # ):
    #     mock_do_embeddings.return_value = False
    #     mock_do_tsne.return_value = True
    #     mock_do_classifier.return_value = False
    #     mock_do_upload.return_value = False
    #     result = self.runner.invoke(
    #         catch_exceptions=False,
    #         cli=run,
    #         args=["--config", "tests/resources/test.yaml"],
    #     )
    #     self.assertEqual(result.exit_code, 0)

    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_upload")
    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    # @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    # @mock.patch("boto3.client")
    # def test_run_do_classifiers(
    #     self,
    #     mock_boto,
    #     mock_do_embeddings,
    #     mock_do_tsne,
    #     mock_do_classifier,
    #     mock_do_upload,
    # ):
    #     mock_do_embeddings.return_value = False
    #     mock_do_tsne.return_value = False
    #     mock_do_classifier.return_value = True
    #     mock_do_upload.return_value = False

    #     result = self.runner.invoke(
    #         catch_exceptions=True,
    #         cli=run,
    #         args=["--config", "tests/resources/test.yaml"],
    #     )

    #     self.assertEqual(result.exit_code, 0)

    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("neat_ml.cli.pre_run_checks")
    def test_run_pre_run_checks(
        self,
        mock_pre_run_checks,
        mock_do_embeddings,
        mock_do_tsne,
        mock_do_classifier,
        mock_do_upload,
    ):
        mock_do_embeddings.return_value = False
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = False
        mock_pre_run_checks.return_value = True
        result = self.runner.invoke(
            catch_exceptions=False,
            cli=run,
            args=["--config", "tests/resources/test.yaml"],
        )
        self.assertTrue(mock_pre_run_checks.called)
        self.assertEqual(result.exit_code, 0)

    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("neat_ml.cli.pre_run_checks")
    def test_run_pre_run_checks(
        self,
        mock_pre_run_checks,
        mock_do_embeddings,
        mock_do_tsne,
        mock_do_classifier,
        mock_do_upload,
    ):
        """Test pre-run checks.

        :param mock_pre_run_checks: Mock param.
        :param mock_do_embeddings: Mock param.
        :param mock_do_tsne: Mock param.
        :param mock_do_classifier: Mock param.
        :param mock_do_upload: Mock param.
        """
        mock_do_embeddings.return_value = False
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = False
        mock_pre_run_checks.return_value = True
        result = self.runner.invoke(
            catch_exceptions=False,
            cli=run,
            args=["--config", "tests/resources/test.yaml"],
        )
        self.assertEqual(result.exit_code, 0)

    @mock.patch("neat_ml.cli.do_update_yaml")
    def test_run_update_yaml(self, mock_do_update_yaml):
        """Test CLI 'updateyaml' command.

        :param mock_do_update_yaml: A mock param.
        """
        result = self.runner.invoke(
            catch_exceptions=False,
            cli=updateyaml,
            args=["--input_path", "tests/resources/test.yaml"],
        )
        self.assertTrue(mock_do_update_yaml.called)
        self.assertEqual(result.exit_code, 0)
