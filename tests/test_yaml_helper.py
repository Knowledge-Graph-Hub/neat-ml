from unittest import TestCase, mock

from grape import Graph  # noqa F401
from parameterized import parameterized

from neat_ml.yaml_helper.yaml_helper import (YamlHelper, download_file, is_url,
                                             is_valid_path, validate_config)


class TestYamlHelper(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_yaml = "tests/resources/test.yaml"
        cls.yh = YamlHelper(cls.test_yaml)
        cls.embedding_args = cls.yh.make_node_embeddings_args()

    def setUp(self) -> None:
        self.test_yaml_upload_bad = "tests/resources/test_bad_upload_info.yaml"

    def test_validate_config(self):
        good_config = {
            "Target": {"target_path": "tests/resources/test_output_data_dir/"},
            "GraphDataConfiguration": {"graph": {"directed": False}},
        }
        bad_config = {
            "Potato": {"target_path": "tests/resources/test_output_data_dir/"},
            "GraphDataConfiguration": {"graph": {"directed": False}},
        }

        self.assertTrue(validate_config(good_config))
        self.assertFalse(validate_config(bad_config))

    def test_outdir(self) -> None:
        self.assertEqual(
            "tests/resources/test_output_data_dir/", self.yh.outdir()
        )

    def test_add_indir_to_graph_data(self):
        # emits error message to log, but continues:
        self.yh.add_indir_to_graph_data(
            graph_data={}, keys_to_add_indir=["not_a_key"]
        )

    def test_do_tsne(self):
        self.assertTrue(hasattr(YamlHelper, "do_tsne"))
        self.assertTrue(self.yh.do_tsne())

    def test_do_embeddings(self):
        self.assertTrue(hasattr(YamlHelper, "do_embeddings"))
        self.assertTrue(self.yh.do_embeddings())

    def test_do_classifier(self):
        self.assertTrue(hasattr(YamlHelper, "do_classifier"))
        self.assertTrue(self.yh.do_classifier())

    def test_do_upload(self):
        self.assertTrue(hasattr(YamlHelper, "do_upload"))
        self.assertTrue(self.yh.do_upload())

    def test_make_upload_args(self):
        self.assertTrue(hasattr(YamlHelper, "make_upload_args"))
        self.assertDictEqual(
            self.yh.make_upload_args(),
            {
                "local_directory": "tests/resources/test_output_data_dir/",
                "s3_bucket": "some_bucket",
                "s3_bucket_dir": "some/remote/directory/",
                "extra_args": {"ACL": "public-read", "BCL": "private-read"},
            },
        )

    def test_classifier_history_file_name(self):
        class_list = self.yh.yaml["ClassifierContainer"]["classifiers"]
        expect_filename = [
            x["history_filename"]
            for x in class_list
            if x["classifier_id"] == "mlp_1"
        ][0]
        self.assertEqual(
            expect_filename,
            "mlp_classifier_history.json",
        )

    @parameterized.expand(
        [
            (
                "main_graph_args",
                {
                    "default_edge_type": "biolink:related_to",
                    "default_node_type": "biolink:NamedThing",
                    "destinations_column": "object",
                    "directed": False,
                    "edge_path": "tests/resources/test_graphs/pos_train_edges.tsv",  # noqa E501
                    "node_path": "tests/resources/test_graphs/pos_train_nodes.tsv",  # noqa E501
                    "node_list_node_types_column": "category",
                    "nodes_column": "id",
                    "sources_column": "subject",
                    "verbose": True,
                },
            ),
            (
                "node_embedding_params",
                {
                    "method_name": "SkipGram",
                    "batch_size": 128,
                    "explore_weight": 1.0,
                    "iterations": 5,
                    "return_weight": 1.0,
                    "walk_length": 10,
                    "window_size": 4,
                },
            ),
            (
                "embedding_outfile",
                "tests/resources/test_output_data_dir/test_embeddings_test_yaml.csv",  # noqa E501
            ),
            (
                "embedding_history_outfile",
                "tests/resources/test_output_data_dir/embedding_history.json",
            ),
            # ("bert_columns", ["category", "id"]),
        ]
    )
    def test_make_embedding_args(self, key, value):
        self.assertTrue(
            key in self.embedding_args,
            msg=f"can't find key {key} in output of make_embedding_args()",
        )
        self.assertEqual(value, self.embedding_args[key])

    @parameterized.expand(
        [
            ("http://foobar.com", True),
            ("https://foobar.com/somefile.tsv", True),
            ("ftp://foobar.com/somefile.tsv", True),
            ("somepath/to/a/file/somefile.tsv", False),
            ("somefile.tsv", False),
        ]
    )
    def test_is_url(self, string, expected_is_url_value):
        self.assertEqual(expected_is_url_value, is_url(string))

    @parameterized.expand(
        [
            ("tests/resources/test_graphs/pos_train_edges.tsv", True),
            ("tests/resources/test_graphs/pos_train_nodes.tsv", True),
            ("s3://bucket/file.tsv", False),
            ("file", False),
            ("pos_train_nodes.***", False),
        ]
    )
    def test_is_valid_path(self, string, expected_value):
        if not expected_value:
            with self.assertRaises(FileNotFoundError):
                is_valid_path(string)
        else:
            self.assertEqual(expected_value, is_valid_path(string))

    @mock.patch("neat_ml.yaml_helper.yaml_helper.download_file")
    @mock.patch("tarfile.open")
    def test_load_graph(self, mock_tarfile_open, mock_download_file):
        self.yh.load_graph()
        self.assertTrue(mock_download_file.called)

    @mock.patch("neat_ml.yaml_helper.yaml_helper.download_file")
    @mock.patch("tarfile.open")
    def test_graph_contains_node_types(
        self, mock_tarfile_open, mock_download_file
    ):
        g = self.yh.load_graph()
        self.assertTrue(mock_download_file.called)
        self.assertEqual(g.get_node_types_number(), 2)
        self.assertCountEqual(
            g.get_unique_node_type_names(), ["biolink:Gene", "biolink:Protein"]
        )

    @mock.patch("neat_ml.yaml_helper.yaml_helper.Request")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.urlopen")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.open")
    def test_download_file(self, mock_open, mock_urlopen, mock_Request):
        download_file("https://someurl.com/file.txt", outfile="someoutfile")
        for this_mock in [mock_open, mock_urlopen, mock_Request]:
            self.assertTrue(this_mock.called)
            self.assertEqual(1, this_mock.call_count)

    @mock.patch("neat_ml.yaml_helper.yaml_helper.Request")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.urlopen")
    @mock.patch("neat_ml.yaml_helper.yaml_helper.open")
    @mock.patch("tarfile.open")
    def test_download_compressed_file(
        self, mock_tarfile_open, mock_open, mock_urlopen, mock_Request
    ):
        download_file("https://someurl.com/file.tar.gz", outfile="file.tar.gz")
        for this_mock in [
            mock_tarfile_open,
            mock_open,
            mock_urlopen,
            mock_Request,
        ]:
            self.assertTrue(this_mock.called)
            self.assertEqual(1, this_mock.call_count)
