import os
from unittest import TestCase

from neat.yaml_helper.yaml_helper import YamlHelper

from neat.graph_embedding.graph_embedding import get_node_data, make_graph_embeddings
import pandas as pd


class TestGraphEmbedding(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.test_node_file = 'tests/resources/test_graphs/test_small_nodes.tsv'
        self.expected_embedding_file = 'output_data/test_embeddings.tsv'
        self.expected_history_file = 'output_data/embedding_history.json'

        if os.path.exists(self.expected_embedding_file):
            print(
                f"removing existing test embedding file {self.expected_embedding_file}")
            os.unlink(self.expected_embedding_file)

    def test_get_node_data(self):
        node_data = get_node_data(self.test_node_file)
        self.assertTrue(isinstance(node_data, pd.DataFrame))

    def test_make_graph_embeddings(self):
        yhelp = YamlHelper("tests/resources/test_graph_embedding_bert_tsne.yaml")
        embed_kwargs = yhelp.make_embedding_args()
        make_graph_embeddings(**embed_kwargs)
        self.assertTrue(os.path.exists(self.expected_embedding_file))
        self.assertTrue(os.path.exists(self.expected_history_file))
