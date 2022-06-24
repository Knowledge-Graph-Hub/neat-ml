"""Test graph embedding."""
import os
from unittest import TestCase

import pandas as pd

from neat_ml.graph_embedding.graph_embedding import (get_node_data,
                                                     make_node_embeddings)


class TestGraphEmbedding(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.test_node_file = (
            "tests/resources/test_graphs/test_small_nodes_with_text.tsv"
        )
        self.expected_embedding_file = "tests/resources/test_output_data_dir/test_embeddings_test_yaml.csv"  # noqa E501
        self.expected_history_file = (
            "tests/resources/test_output_data_dir/embedding_history.json"
        )
        self.node_embed_args = {
            "embedding_outfile": "tests/resources/test_output_data_dir/test_embeddings_test_yaml.csv",  # noqa E501
            "embedding_history_outfile": "tests/resources/test_output_data_dir/embedding_history.json",  # noqa E501
            "main_graph_args": {
                "directed": False,
                "node_path": "tests/resources/test_graphs/test_small_nodes_with_text.tsv",  # noqa E501
                "edge_path": "tests/resources/test_graphs/test_small_edges.tsv",  # noqa E501
                "verbose": True,
                "nodes_column": "id",
                "node_list_node_types_column": "category",
                "default_node_type": "biolink:NamedThing",
                "sources_column": "subject",
                "destinations_column": "object",
                "default_edge_type": "biolink:related_to",
            },
            "node_embedding_params": {
                "method_name": "SkipGram",
                "walk_length": 4,
                "batch_size": 8,
                "window_size": 2,
                "return_weight": 1.0,
                "explore_weight": 1.0,
                "iterations": 2,
            },
            "bert_columns": {"description": ""},
        }

        if os.path.exists(self.expected_embedding_file):
            print(
                f"removing existing test embedding \
                  file {self.expected_embedding_file}"
            )
            os.unlink(self.expected_embedding_file)

    def test_get_node_data(self):
        node_data = get_node_data(self.test_node_file)
        self.assertTrue(isinstance(node_data, pd.DataFrame))

    def test_make_graph_embeddings(self):

        node_embedding_args = self.node_embed_args
        make_node_embeddings(**node_embedding_args)
        self.assertTrue(os.path.exists(self.expected_embedding_file))
