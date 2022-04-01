from unittest import TestCase, skip, mock
from parameterized import parameterized

from neat.yaml_helper.yaml_helper import YamlHelper, catch_keyerror, is_url, \
    download_file, is_valid_path
import os

from ensmallen import Graph  # type: ignore

class TestYamlHelper(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_yaml = "tests/resources/test.yaml"
        cls.yh = YamlHelper(cls.test_yaml)
        cls.embedding_args = cls.yh.make_node_embeddings_args()

    def setUp(self) -> None:
        self.test_yaml_upload_good = 'tests/resources/test_good_upload_info.yaml'
        self.test_yaml_upload_bad = 'tests/resources/test_bad_upload_info.yaml'
        self.test_yaml_bert_tsne = 'tests/resources/test_graph_embedding_bert_tsne.yaml'

    def test_no_indir(self) -> None:
        yh = YamlHelper("tests/resources/test_no_indir.yaml")
        self.assertEqual("", yh.indir())

    def test_bad_indir(self) -> None:
        with self.assertRaises(FileNotFoundError) as context:
            YamlHelper("tests/resources/test_bad_indir.yaml").indir()

    def test_outdir(self) -> None:
        self.assertEqual("output_data", self.yh.outdir())

    def test_add_indir_to_graph_data(self):
        # emits error message to log, but continues:
        self.yh.add_indir_to_graph_data(graph_data={}, keys_to_add_indir=['not_a_key'])

    def test_do_tsne(self):
        self.assertTrue(hasattr(YamlHelper, 'do_tsne'))
        self.assertTrue(not self.yh.do_tsne())
        ybt = YamlHelper(self.test_yaml_bert_tsne)
        self.assertTrue(ybt.do_tsne())

    def test_do_embeddings(self):
        self.assertTrue(hasattr(YamlHelper, 'do_embeddings'))
        self.assertTrue(self.yh.do_embeddings())

    def test_do_classifier(self):
        self.assertTrue(hasattr(YamlHelper, 'do_classifier'))
        self.assertTrue(self.yh.do_classifier())

    def test_do_upload(self):
        self.assertTrue(hasattr(YamlHelper, 'do_upload'))
        yg = YamlHelper(self.test_yaml_upload_good)
        self.assertTrue(yg.do_upload())

    def test_make_upload_args(self):
        self.assertTrue(hasattr(YamlHelper, 'make_upload_args'))
        yg = YamlHelper(self.test_yaml_upload_good)
        self.assertDictEqual(
            yg.make_upload_args(),
            {'local_directory': 'tests/resources/test_output_data_dir/',
             's3_bucket': 'some_bucket', 's3_bucket_dir': 'some/remote/directory/',
             'extra_args': {'ACL': 'public-read'}})

    def test_classifier_history_file_name(self):
        self.assertTrue(hasattr(YamlHelper, 'classifier_history_file_name'))
        yg = YamlHelper(self.test_yaml)
        self.assertEqual(
            yg.classifier_history_file_name(yg.yaml['classifier']['classifiers'][0]),
            "mlp_classifier_history.json")

    @parameterized.expand([
        ('main_graph_args', {'default_edge_type': 'biolink:related_to',
                             'default_node_type': 'biolink:NamedThing',
                             'destinations_column': 'object',
                             'directed': False,
                             'edge_path': 'tests/resources/test_graphs/pos_train_edges.tsv',
                             'node_path': 'tests/resources/test_graphs/pos_train_nodes.tsv',
                             'node_list_node_types_column': 'category',
                             'nodes_column': 'id',
                             'sources_column': 'subject',
                             'verbose': True}),
        ('node_embedding_params', {
            'node_embedding_method_name': 'SkipGram',
            'batch_size': 128,
            'explore_weight': 1.0,
            'iterations': 5,
            'return_weight': 1.0,
            'walk_length': 10,
            'window_size': 4}),
        ('embedding_outfile', 'output_data/test_embeddings_test_yaml.tsv'),
        ('embedding_history_outfile', 'output_data/embedding_history.json'),
        ('bert_columns', ['category', 'id']),
    ])
    def test_make_embedding_args(self, key, value):
        self.assertTrue(key in self.embedding_args,
                        msg=f"can't find key {key} in output of make_embedding_args()")
        self.assertEqual(value, self.embedding_args[key])

    def test_make_embeddings_metrics_class_list(self):
        self.assertTrue(hasattr(YamlHelper, 'make_embeddings_metrics_class_list'))
        yh = YamlHelper("tests/resources/test_make_embeddings_metrics.yaml")
        cl = yh.make_embeddings_metrics_class_list()
        self.assertEqual(3, len(cl))
        self.assertCountEqual(["<class 'keras.metrics.AUC'>",
                               "<class 'keras.metrics.Recall'>",
                               "<class 'keras.metrics.Precision'>"],
                              [str(klass.__class__) for klass in cl])

    def test_catch_keyerror(self):
        yh = YamlHelper("tests/resources/test_no_graph.yaml")
        yh.pos_val_graph_args()  # no assertion needed, just testing for no exception

    @parameterized.expand([
        ('http://foobar.com', True),
        ('https://foobar.com/somefile.tsv', True),
        ('ftp://foobar.com/somefile.tsv', True),
        ('somepath/to/a/file/somefile.tsv', False),
        ('somefile.tsv', False),
    ])
    def test_is_url(self, string, expected_is_url_value):
        self.assertEqual(expected_is_url_value, is_url(string))

    @parameterized.expand([
        ('tests/resources/test_graphs/pos_train_edges.tsv', True),
        ('tests/resources/test_graphs/pos_train_nodes.tsv', True),
        ('s3://bucket/file.tsv', False),
        ('file', False),
        ('pos_train_nodes.***', False),
    ])
    def test_is_valid_path(self, string, expected_value):
        if not expected_value:
             with self.assertRaises(FileNotFoundError):
                 is_valid_path(string)
        else:
            self.assertEqual(expected_value, is_valid_path(string))

    def test_load_graph(self):
        self.yh.load_graph()
        # No assertion here - this will error if it fails

    def test_graph_contains_node_types(self):
        g = self.yh.load_graph()
    
        self.assertEqual(g.get_node_types_number(), 2)
        self.assertCountEqual(g.get_unique_node_type_names(), 
                                ['biolink:Gene', 'biolink:Protein'])

    @mock.patch('neat.yaml_helper.yaml_helper.download_file')
    @mock.patch('ensmallen.Graph.from_csv')
    def test_node_edge_urls_converted_to_path(self, mock_from_csv, mock_download_file):
        this_yh = YamlHelper('tests/resources/test_urls_for_node_and_edge_paths.yaml')
        self.assertTrue(is_url(this_yh.main_graph_args()['node_path']))
        self.assertTrue(is_url(this_yh.main_graph_args()['edge_path']))

        this_yh.load_graph()

        self.assertFalse(is_url(this_yh.main_graph_args()['node_path']))
        self.assertEqual('output_data/https___someremoteurl.com_nodes.tsv',
                         this_yh.main_graph_args()['node_path'])
        self.assertFalse(is_url(this_yh.main_graph_args()['edge_path']))
        self.assertEqual('output_data/https___someremoteurl.com_edges.tsv',
                         this_yh.main_graph_args()['edge_path'])

    @mock.patch('neat.yaml_helper.yaml_helper.download_file')
    @mock.patch('ensmallen.Graph.from_csv')
    def test_node_edge_urls_file_downloaded(self, mock_from_csv, mock_download_file):
        this_yh = YamlHelper('tests/resources/test_urls_for_node_and_edge_paths.yaml')
        this_yh.load_graph()
        self.assertTrue(mock_download_file.called)
        self.assertEqual(2, mock_download_file.call_count)

    @mock.patch('neat.yaml_helper.yaml_helper.download_file')
    def test_embeddings_url_converted_to_path(self, mock_download_file):
        this_yh = YamlHelper('tests/resources/test_url_for_embeddings.yaml')
        self.assertFalse(is_url(this_yh.embedding_outfile()))
        self.assertEqual('output_data/https___someremoteurl.com_embeddings.tsv',
                         this_yh.embedding_outfile())

    @mock.patch('neat.yaml_helper.yaml_helper.download_file')
    def test_embeddings_url_file_downloaded(self, mock_download_file):
        this_yh = YamlHelper('tests/resources/test_url_for_embeddings.yaml')
        this_yh.embedding_outfile()
        self.assertTrue(mock_download_file.called)

    @mock.patch('neat.yaml_helper.yaml_helper.Request')
    @mock.patch('neat.yaml_helper.yaml_helper.urlopen')
    @mock.patch('neat.yaml_helper.yaml_helper.open')
    def test_download_file(self, mock_open, mock_urlopen, mock_Request):
        download_file("https://someurl.com/file.txt", outfile="someoutfile")
        for this_mock in [mock_open, mock_urlopen, mock_Request]:
            self.assertTrue(this_mock.called)
            self.assertEqual(1, this_mock.call_count)
