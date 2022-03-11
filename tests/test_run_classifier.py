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
        cls.test_embeddings = pd.read_csv(
            "tests/resources/test_embeddings.tsv", header=None
        )
        # cls.model = pickle.load(
        #     open(
        #         "tests/resources/test_output_data_dir/model_lr_test_yaml.h5",
        #         "rb",
        #     )
        # )

    def setUp(self) -> None:
        pass

    def test_reality(self):
        self.assertEqual(1, 1)

    def test_run_classifier(self):
        # temp = tempfile.NamedTemporaryFile().name  # once we have test firmed up
        outfile = "/dev/null"
        # outfile = os.path.join(dirname(__file__), "resources/tmp/test.tsv")
        with open('tests/resources/model_lr_test_yaml.h5', 'rb') as f:
            m = pickle.load(f)
            predict_links(
                graph=self.graph,
                model=m,
                node_types=[["biolink:Gene"], ["biolink:Protein"]],
                cutoff=0.8,
                output_file=outfile,
                embeddings=self.test_embeddings,
                edge_method="foo",
                verbose=False,
            )
