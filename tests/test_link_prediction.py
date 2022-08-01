"""Test link prediction."""
import os
import pathlib
from unittest import TestCase

import numpy as np
import pandas as pd
from grape import Graph

try:
    from keras.engine.sequential import Sequential

    HAVE_KERAS = True
except ModuleNotFoundError:
    print("Keras not found - will not test related functions.")
    HAVE_KERAS = False

from neat_ml.link_prediction.grape_model import GrapeModel
from neat_ml.link_prediction.mlp_model import MLPModel
from neat_ml.link_prediction.sklearn_model import SklearnModel
from neat_ml.run_classifier.run_classifier import get_custom_model_path
from neat_ml.yaml_helper.yaml_helper import YamlHelper


class TestLinkPrediction(TestCase):
    """Test link prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up."""
        pass

    def setUp(self) -> None:
        """Set up."""
        self.yaml_file = "tests/resources/test.yaml"
        self.embed_file = "tests/resources/test_link_prediction/test_embeddings_test_yaml.csv"  # noqa E501
        self.embed_snippet_file = "tests/resources/test_link_prediction/test_embeddings_test_yaml_SNIPPET.csv"  # noqa E501
        self.yhelp = YamlHelper(self.yaml_file)
        self.test_model_path = "tests/resources/test_output_data_dir/"
        self.test_load_path = "tests/resources/test_link_prediction/"

        self.sklearn_model = SklearnModel(
            (self.yhelp.classifiers())[0], self.test_model_path
        )
        self.tf_model = MLPModel(
            (self.yhelp.classifiers())[1], self.test_model_path
        )
        self.grape_model = GrapeModel(
            (self.yhelp.classifiers())[2], self.test_model_path
        )

        self.sklearn_outfile = ((self.yhelp.classifiers())[0])["outfile"]

        self.generic_tf_outfile = ((self.yhelp.classifiers())[1])["outfile"]
        self.custom_tf_outfile = get_custom_model_path(self.generic_tf_outfile)

        self.grape_outfile = ((self.yhelp.classifiers())[2])["outfile"]

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

    def assert_is_file(self, path):
        """Assert if path is a file."""
        if not pathlib.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_sklearn_save(self) -> None:
        """Test saving model using sklearn."""
        model_object = self.sklearn_model

        # Need to have a fitted model here
        embed_contents = pd.read_csv(
            self.embed_snippet_file, index_col=0, header=None
        )

        dummy_labels = np.random.randint(
            0, high=2, size=(embed_contents.shape[0],), dtype=np.bool
        )

        model_object.fit(embed_contents, dummy_labels)

        model_object.save()

        self.assert_is_file(
            os.path.join(self.test_model_path, self.sklearn_outfile)
        )

    def test_tf_save(self) -> None:
        """Test saving model in tensorflow."""
        model_object = self.tf_model

        model_object.save()

        self.assert_is_file(
            os.path.join(self.test_model_path, self.generic_tf_outfile)
        )
        self.assert_is_file(
            os.path.join(self.test_model_path, self.custom_tf_outfile)
        )

    # Note that the load tests *do not* use the files created by
    # the save tests above, so they may remain independent.

    def test_sklearn_load(self) -> None:
        """Test sklearn loading."""
        out_fn = os.path.join(self.test_load_path, self.sklearn_outfile)
        model_object = self.sklearn_model.load(out_fn)
        self.assertEqual(type(model_object), SklearnModel)

    def test_sklearn_fit(self) -> None:
        """Test sklearn fitting."""
        model_object = self.sklearn_model
        result, _ = model_object.make_train_valid_data(
            embedding_file=self.embed_file,
            training_graph_args=self.training_graph_args,
            edge_method="Average",
        )

        fit_out = model_object.fit(*result)
        self.assertEqual(str(fit_out), "LogisticRegression()")

    def test_tf_load(self) -> None:
        """Test tensorflow load."""
        out_fn = os.path.join(self.test_load_path, self.generic_tf_outfile)
        (
            generic_model_object,
            customized_model_object,
        ) = self.tf_model.load(out_fn)

        if HAVE_KERAS:
            self.assertEqual(type(generic_model_object), Sequential)
        self.assertEqual(type(customized_model_object), MLPModel)

    def test_sklearn_make_link_prediction_data(self) -> None:
        """Test sklearn link predication data generation."""
        model_object = self.sklearn_model
        result = model_object.make_train_valid_data(
            embedding_file=self.embed_file,
            training_graph_args=self.training_graph_args,
            edge_method="Average",
        )
        # result contains tuple of tuples of 2-dim arrays
        self.assertEqual(result[0][0].ndim, 2)

    def test_grape_fit(self) -> None:
        """Test grape's Ensmallen model fitting."""
        model_object = self.grape_model
        graph_in = Graph.from_csv(**self.training_graph_args)
        model_object.fit(graph_in)
        self.assertTrue(model_object.is_fit)
        output = model_object.predict_proba(graph_in)
        self.assertGreaterEqual(len(output), 470000)
