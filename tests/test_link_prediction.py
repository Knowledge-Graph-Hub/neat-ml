import os
import pathlib
from unittest import TestCase

import numpy as np
import pandas as pd
from keras.engine.sequential import Sequential

from neat_ml.link_prediction.mlp_model import MLPModel
from neat_ml.link_prediction.sklearn_model import SklearnModel
from neat_ml.run_classifier.run_classifier import get_custom_model_path
from neat_ml.yaml_helper.yaml_helper import YamlHelper


class TestLinkPrediction(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.yaml_file_sklearn = "tests/resources/test.yaml"
        cls.yaml_file_tf = "tests/resources/test.yaml"
        cls.embed_file = (
            "tests/resources/test_link_prediction/test_embeddings_test_yaml.csv"
        )
        cls.yhelp_sklearn = YamlHelper(cls.yaml_file_sklearn)
        cls.yhelp_tf = YamlHelper(cls.yaml_file_tf)
        cls.test_model_path = "tests/resources/test_output_data_dir/"
        cls.test_load_path = "tests/resources/test_link_prediction/"
        cls.sklearn_model = SklearnModel(
            (cls.yhelp_sklearn.classifiers())[0], cls.test_model_path
        )
        cls.tf_model = MLPModel((cls.yhelp_tf.classifiers())[1], cls.test_model_path)
        cls.sklearn_outfile = ((cls.yhelp_sklearn.classifiers())[0])["outfile"]
        cls.generic_tf_outfile = ((cls.yhelp_tf.classifiers())[1])["outfile"]
        cls.custom_tf_outfile = get_custom_model_path(cls.generic_tf_outfile)
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

    def setUp(self) -> None:
        pass

    def assertIsFile(self, path):
        if not pathlib.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_sklearn_save(self) -> None:
        model_object = self.sklearn_model

        # Need to have a fitted model here
        embed_contents = pd.read_csv(self.embed_file, index_col=0, header=None)

        dummy_labels = np.random.randint(
            0, high=2, size=(embed_contents.shape[0],), dtype=np.bool
        )

        model_object.fit(embed_contents, dummy_labels)

        model_object.save()

        self.assertIsFile(os.path.join(self.test_model_path, self.sklearn_outfile))

    def test_tf_save(self) -> None:
        model_object = self.tf_model

        model_object.save()

        self.assertIsFile(os.path.join(self.test_model_path, self.generic_tf_outfile))
        self.assertIsFile(os.path.join(self.test_model_path, self.custom_tf_outfile))

    # Note that the load tests *do not* use the files created by
    # the save tests above, so they may remain independent.

    def test_sklearn_load(self) -> None:
        out_fn = os.path.join(self.test_load_path, self.sklearn_outfile)
        model_object = self.sklearn_model.load(out_fn)
        self.assertEqual(type(model_object), SklearnModel)

    def test_sklearn_fit(self) -> None:
        model_object = self.sklearn_model
        result, _ = model_object.make_train_valid_data(
            embedding_file=self.embed_file,
            training_graph_args=self.training_graph_args,
            edge_method="Average",
        )

        fit_out = model_object.fit(*result)
        self.assertEqual(str(fit_out), "LogisticRegression()")

    def test_tf_load(self) -> None:
        out_fn = os.path.join(self.test_load_path, self.generic_tf_outfile)
        (
            generic_model_object,
            customized_model_object,
        ) = self.tf_model.load(out_fn)
        self.assertEqual(type(generic_model_object), Sequential)
        self.assertEqual(type(customized_model_object), MLPModel)

    def test_sklearn_make_link_prediction_data(self) -> None:
        model_object = self.sklearn_model
        result = model_object.make_train_valid_data(
            embedding_file=self.embed_file,
            training_graph_args=self.training_graph_args,
            edge_method="Average",
        )
        # result contains tuple of tuples of 2-dim arrays
        self.assertEqual(result[0][0].ndim, 2)
