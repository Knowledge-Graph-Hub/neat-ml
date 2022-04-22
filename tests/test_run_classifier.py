import os
import pickle
from posixpath import dirname
import tempfile
from unittest import TestCase, skip
from neat.run_classifier.run_classifier import predict_links
from neat.yaml_helper.yaml_helper import YamlHelper
from ensmallen import Graph
import pandas as pd


class TestRunClassifier(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.yaml_file = "tests/resources/test.yaml"
        cls.yhelp = YamlHelper(cls.yaml_file)
        cls.graph = Graph.from_csv(**cls.yhelp.main_graph_args())
        cls.test_embeddings = (
            "tests/resources/test_run_classifier/test_embeddings_test_yaml.csv"
        )
        cls.test_model_path = (
            "tests/resources/test_run_classifier/model_lr_test_yaml.h5"
        )
        cls.training_graph_args = {
            "directed": False,
            "node_path": "tests/resources/test_graphs/pos_train_nodes.tsv",
            "edge_path": "tests/resources/test_graphs/pos_train_edges.tsv",
            "verbose": True,
            "nodes_column": "id",
            "node_list_node_types_column": "category",
            "default_node_type": "biolink:NamedThing",
            "sources_column": "subject",
            "destinations_column": "object",
            "default_edge_type": "biolink:related_to",
        }
        cls.negative_graph_args = {
            "directed": False,
            "edge_path": "tests/resources/test_graphs/neg_train_edges.tsv",
        }

    def setUp(self) -> None:
        pass

    def test_reality(self):
        self.assertEqual(1, 1)

    def test_run_classifier(self):
        # temp = tempfile.NamedTemporaryFile().name  # once we have test firmed up
        # outfile = "/dev/null"
        outdir = "tests/resources/tmp/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        outfile = os.path.join(dirname(__file__), "resources/tmp/test.tsv")
        with open(self.test_model_path, "rb") as f:
            m = pickle.load(f)

        predict_links(
            graph=self.graph,
            model=m,
            node_types=[["biolink:Gene"], ["biolink:Protein"]],
            cutoff=0.8,
            output_file=outfile,
            embeddings_file=self.test_embeddings,
            edge_method="Average",
            verbose=True,
        )
