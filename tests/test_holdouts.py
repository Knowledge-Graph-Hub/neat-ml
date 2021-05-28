import os
import tempfile
from unittest import TestCase

from neat.holdouts.holdouts import make_holdouts
from neat.yaml_helper.yaml_helper import YamlHelper


class TestHoldouts(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_yaml_file = 'tests/resources/test.yaml'
        cls.yaml_helper = YamlHelper(cls.test_yaml_file)
        cls.main_graph_args = cls.yaml_helper.main_graph_args()
        cls.train_fraction = 0.99
        cls.seed = 21
        cls.tmp_output_dir = tempfile.TemporaryDirectory()
        # expensive, do this once:
        make_holdouts(main_graph_args=cls.main_graph_args,
                      output_dir=cls.tmp_output_dir.name,
                      train_size=cls.train_fraction, validation=False, seed=cls.seed)

    def setUp(self) -> None:
        pass

    def test_output_files(self):
        self.assertCountEqual(os.listdir(self.tmp_output_dir.name),
                             ['pos_train_edges.tsv', 'neg_train_edges.tsv',
                              'pos_test_edges.tsv', 'pos_train_nodes.tsv',
                              'neg_test_edges.tsv'])

    def test_validation_files(self):
        different_tmpdir = tempfile.TemporaryDirectory()
        make_holdouts(main_graph_args=self.main_graph_args,
                      output_dir=different_tmpdir.name,
                      train_size=self.train_fraction, validation=True, seed=self.seed)
        self.assertCountEqual(os.listdir(different_tmpdir.name),
                             ['neg_valid_edges.tsv', 'pos_valid_edges.tsv',
                              'pos_train_edges.tsv', 'neg_train_edges.tsv',
                              'pos_test_edges.tsv', 'pos_train_nodes.tsv',
                              'neg_test_edges.tsv'])

