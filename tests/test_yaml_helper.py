import math
from unittest import TestCase

from parameterized import parameterized

from neat.yaml_helper.yaml_helper import YamlHelper


class TestYamlHelper(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_yaml = "tests/resources/test.yaml"
        cls.yh = YamlHelper(cls.test_yaml)
        cls.embedding_args = cls.yh.make_embedding_args()

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

    @parameterized.expand([
        ('main_graph_args', {'default_edge_type': 'biolink:related_to',
             'default_node_type': 'biolink:NamedThing',
             'destinations_column': 'object',
             'directed': False,
             'edge_path': 'tests/resources/test_graphs/pos_train_edges.tsv',
             'ignore_duplicated_edges': True,
             'ignore_duplicated_nodes': True,
             'node_path': 'tests/resources/test_graphs/pos_train_nodes.tsv',
             'node_types_column': 'category',
             'nodes_column': 'id',
             'skip_self_loops': False,
             'sources_column': 'subject',
             'verbose': True}),
        ('pos_valid_graph_args', {'edge_path':
                                  'tests/resources/test_graphs/pos_valid_edges.tsv'}),
        ('embiggen_seq_args', {'batch_size': 128,
                         'explore_weight': 1.0,
                         'iterations': 5,
                         'return_weight': 1.0,
                         'walk_length': 10,
                         'window_size': 4}),
        ('node2vec_params', {'embedding_size': 100, 'negative_samples': 30}),
        ('epochs', 1),
        ('early_stopping_args', {'min_delta': 0.0001,
             'monitor': 'loss',
             'patience': 5,
             'restore_best_weights': True}),
        ('model', 'skipgram'),
        ('embedding_outfile', 'output_data/test_embeddings_test_yaml.tsv'),
        ('model_outfile', 'output_data/embedding_model_test_yaml.h5'),
        ('embedding_history_outfile', 'output_data/embedding_history.json'),
        ('use_pos_valid_for_early_stopping', False),
        ('learning_rate', 0.1),
        ('bert_columns', ['category', 'id']),
    ])
    def test_make_embedding_args(self, key, value):
        self.assertTrue(key in self.embedding_args,
                        msg=f"can't find key {key} in output of make_embedding_args()")
        self.assertEqual(self.embedding_args[key], value)

