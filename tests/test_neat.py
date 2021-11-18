import re
from unittest import TestCase, mock
from click.testing import CliRunner

from neat.cli import run


class TestRun(TestCase):
    """Tests the neat.py script."""
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_run_no_yaml_file(self):
        result = self.runner.invoke(catch_exceptions=False,
                                    cli=run,
                                    args=['--config', 'doesntexist'])
        self.assertTrue(re.search('doesntexist', result.output))
        self.assertNotEqual(result.exit_code, 0)

    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("boto3.client")
    def test_run_do_upload(self, mock_boto3_client,
                           mock_do_embeddings, mock_do_tsne, mock_do_classifier,
                           mock_do_upload):
        mock_do_embeddings.return_value = False
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = True
        result = self.runner.invoke(catch_exceptions=False,
                                    cli=run,
                                    args=['--config',
                                          'tests/resources/test_neat.yaml'])
        self.assertTrue(mock_do_upload.called)
        self.assertTrue(mock_boto3_client.called)
        self.assertEqual(result.exit_code, 0)

    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("neat.graph_embedding.graph_embedding.make_node_embeddings")
    def test_run_do_embeddings(self, mock_make_node_embeddings,
                               mock_do_embeddings, mock_do_tsne, mock_do_classifier,
                               mock_do_upload):
        mock_do_embeddings.return_value = True
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = False
        result = self.runner.invoke(catch_exceptions=False,
                                    cli=run,
                                    args=['--config',
                                          'tests/resources/test_neat.yaml'])
        self.assertEqual(result.exit_code, 0)

    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    @mock.patch("neat.visualization.visualization.make_tsne")
    def test_run_do_embeddings(self, mock_make_tnse,
                               mock_do_embeddings, mock_do_tsne, mock_do_classifier,
                               mock_do_upload):
        mock_do_embeddings.return_value = False
        mock_do_tsne.return_value = True
        mock_do_classifier.return_value = False
        mock_do_upload.return_value = False
        result = self.runner.invoke(catch_exceptions=False,
                                    cli=run,
                                    args=['--config',
                                          'tests/resources/test_neat.yaml'])
        self.assertEqual(result.exit_code, 0)

    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_upload")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_classifier")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_tsne")
    @mock.patch("neat.yaml_helper.yaml_helper.YamlHelper.do_embeddings")
    def test_run_do_classifiers(self,
                               mock_do_embeddings, mock_do_tsne, mock_do_classifier,
                               mock_do_upload):
        mock_do_embeddings.return_value = False
        mock_do_tsne.return_value = False
        mock_do_classifier.return_value = True
        mock_do_upload.return_value = False
        result = self.runner.invoke(catch_exceptions=False,
                                    cli=run,
                                    args=['--config',
                                          'tests/resources/test_neat.yaml'])
        self.assertEqual(result.exit_code, 0)
