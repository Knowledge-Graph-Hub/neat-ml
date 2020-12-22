from unittest import TestCase

from neat.yaml_helper.yaml_helper import YamlHelper

from neat.graph_embedding.graph_embedding import get_node_data, make_graph_embeddings
import pandas as pd

from neat.yaml_helper import yaml_helper


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
        yhelp = YamlHelper("tests/resources/test_graph_embedding.yaml")
        embed_kwargs = yhelp.make_embedding_args()
        make_graph_embeddings(**embed_kwargs)
