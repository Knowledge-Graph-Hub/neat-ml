import os
import pathlib
from unittest import TestCase, skip

from neat.link_prediction import model
from neat.link_prediction.sklearn_model import SklearnModel
from neat.yaml_helper.yaml_helper import YamlHelper

from sklearn.linear_model._logistic import LogisticRegression

import numpy as np

class TestLinkPrediction(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.yaml_file = "tests/resources/test_neat.yaml"
        cls.embed_file = "tests/resources/test_link_prediction/test_embeddings_test_yaml.csv"
        cls.yhelp = YamlHelper(cls.yaml_file)
        cls.test_model_path = "tests/resources/test_link_prediction/"
        cls.sklearn_model = SklearnModel(
            (cls.yhelp.classifiers())[0], cls.test_model_path
        )
        cls.generic_outfile = ((cls.yhelp.classifiers())[0])["model"][
            "outfile"
        ]
        fn, ext = cls.generic_outfile.split(".")
        cls.custom_outfile = fn + "_custom." + ext
        cls.training_graph_args = {"directed": False,
                                    "node_path": 'tests/resources/test_graphs/pos_train_nodes.tsv',
                                    "edge_path": 'tests/resources/test_graphs/pos_train_edges.tsv',
                                    "verbose": True,
                                    "nodes_column": 'id',
                                    "node_list_node_types_column": 'category',
                                    "default_node_type": 'biolink:NamedThing',
                                    "sources_column": 'subject',
                                    "destinations_column": 'object',
                                    "default_edge_type": 'biolink:related_to'}

    def setUp(self) -> None:
        pass

    def assertIsFile(self, path):
        if not pathlib.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_sklearn_save(self) -> None:
        model_object = self.sklearn_model
        model_object.save()

        self.assertIsFile(
            os.path.join(self.test_model_path, self.generic_outfile)
        )
        self.assertIsFile(
            os.path.join(self.test_model_path, self.custom_outfile)
        )

    def test_sklearn_load(self) -> None:
        out_fn = os.path.join(self.test_model_path, self.generic_outfile)
        (
            generic_model_object,
            customized_model_object,
        ) = self.sklearn_model.load(out_fn)
        self.assertEqual(type(generic_model_object), LogisticRegression)
        self.assertEqual(type(customized_model_object), SklearnModel)

    def test_make_link_prediction_data(self) -> None:
        model_object = self.sklearn_model
        result = model_object.make_link_prediction_data(
                    embedding_file=self.embed_file,
                    training_graph_args=self.training_graph_args,
                    edge_method="Average",
        )
        # result contains tuple of tuples of 2-dim arrays
        self.assertEqual(result[0][0].ndim, 2)
