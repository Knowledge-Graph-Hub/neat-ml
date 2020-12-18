from unittest import TestCase

from neat.graph_embedding.graph_embedding import get_node_data, make_graph_embeddings
import pandas as pd


class TestGraphEmbedding(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.test_node_file = 'tests/resources/test_graphs/test_small_nodes.tsv'

    def test_get_node_data(self):
        node_data = get_node_data(self.test_node_file)
        self.assertTrue(isinstance(node_data, pd.DataFrame))

    def test_make_graph_embeddings(self):
        make_graph_embeddings
        self.assertTrue(True)
