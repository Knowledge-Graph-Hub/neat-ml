"""Test run classifier."""
import os
import pickle  # noqa S403
from posixpath import dirname
from unittest import TestCase

from grape import Graph

from neat_ml.link_prediction.grape_model import GrapeModel
from neat_ml.run_classifier.run_classifier import predict_links
from neat_ml.yaml_helper.yaml_helper import YamlHelper


class TestRunClassifier(TestCase):
    """Test run classifier."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up."""
        pass

    def setUp(self) -> None:
        """Set up."""
        self.yaml_file = "tests/resources/test.yaml"
        self.yhelp = YamlHelper(self.yaml_file)
        self.graph = Graph.from_csv(**self.yhelp.main_graph_args())
        self.test_embeddings = (
            "tests/resources/test_run_classifier/test_embeddings_test_yaml.csv"
        )
        self.test_model_path = (
            "tests/resources/test_run_classifier/model_lr_test_yaml.h5"
        )
        self.test_alt_path = (
            "tests/resources/test_run_classifier/model_pt_test_yaml"
        )
        self.training_graph_args = {
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
        self.negative_graph_args = {
            "directed": False,
            "edge_path": "tests/resources/test_graphs/neg_train_edges.tsv",
        }

    def test_reality(self):
        """Test reality."""
        self.assertEqual(1, 1)

    def test_run_classifier(self):
        """Test run classifier."""
        # temp = tempfile.NamedTemporaryFile().name
        # once we have test firmed up
        # outfile = "/dev/null"
        outdir = "tests/resources/tmp/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        with open(self.test_model_path, "rb") as f:
            m = pickle.load(f)  # noqa S301

        outfile = os.path.join(dirname(__file__), "resources/tmp/test.tsv")
        predict_links(
            graph=self.graph,
            model=m,
            node_types=None,  # No filter
            cutoff=0.0001,
            output_file=outfile,
            embeddings_file=self.test_embeddings,
            edge_method="Average",
            verbose=True,
        )
        with open(outfile) as f:
            self.assertGreater(
                len(f.readlines()),
                9000,
                "Link prediction output is too short.",
            )

        # This file should just have the header
        outfile = os.path.join(dirname(__file__), "resources/tmp/test2.tsv")
        predict_links(
            graph=self.graph,
            model=m,
            node_types=None,  # No filter
            cutoff=0.9,
            output_file=outfile,
            embeddings_file=self.test_embeddings,
            edge_method="Average",
            verbose=True,
        )
        with open(outfile) as f:
            self.assertEqual(1, len(f.readlines()))

    def test_run_classifier_with_node_filters(self):
        """Test run classifier with node filters."""
        outdir = "tests/resources/tmp/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        with open(self.test_model_path, "rb") as f:
            m = pickle.load(f)  # noqa S301

        outfile = os.path.join(
            dirname(__file__), "resources/tmp/test_node_filt.tsv"
        )
        predict_links(
            graph=self.graph,
            model=m,
            node_types=[
                ["biolink:Gene", "biolink:Protein"],
                ["biolink:Protein"],
            ],
            cutoff=0.0001,
            output_file=outfile,
            embeddings_file=self.test_embeddings,
            edge_method="Average",
            verbose=True,
        )
        with open(outfile) as f:
            self.assertGreater(
                len(f.readlines()),
                3000,
                "Link prediction output is too short.",
            )

    def test_grape_link_predict(self) -> None:
        """Test grape's Ensmallen model edge prediction."""
        grape_model = GrapeModel(
            (self.yhelp.classifiers())[2], self.test_alt_path
        )
        graph_in = Graph.from_csv(**self.training_graph_args)
        grape_model.fit(graph_in)
        output = grape_model.predict_proba(graph_in)
        self.assertGreaterEqual(len(output), 470000)
