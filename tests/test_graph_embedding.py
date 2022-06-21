import json
import os
from unittest import TestCase

import pandas as pd

from neat_ml.graph_embedding.graph_embedding import (get_node_data,
                                                     make_node_embeddings)
from neat_ml.yaml_helper.yaml_helper import YamlHelper

# class TestGraphEmbedding(TestCase):

#     @classmethod
#     def setUpClass(cls) -> None:
#         pass

#     def setUp(self) -> None:
#         self.test_node_file = 'tests/resources/test_graphs/test_small_nodes.tsv'
#         self.expected_embedding_file = 'output_data/test_embeddings.tsv'
#         self.expected_history_file = 'output_data/embedding_history.json'

#         if os.path.exists(self.expected_embedding_file):
#             print(
#                 f"removing existing test embedding file {self.expected_embedding_file}")
#             os.unlink(self.expected_embedding_file)

#     def test_get_node_data(self):
#         node_data = get_node_data(self.test_node_file)
#         self.assertTrue(isinstance(node_data, pd.DataFrame))

# def test_make_graph_embeddings(self):
#     yhelp = YamlHelper("tests/resources/test_graph_embedding_bert_tsne.yaml")
#     node_embedding_args = yhelp.make_node_embeddings_args()
#     make_node_embeddings(**node_embedding_args)
#     self.assertTrue(os.path.exists(self.expected_embedding_file))

#     self.assertTrue(os.path.exists(self.expected_history_file))
#     with open(self.expected_history_file) as f:
#         data = f.read()
#         obj = json.loads(data)
#         self.assertListEqual(list(obj.keys()),
#                              ['loss', 'lr'])
