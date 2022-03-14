import os
import pathlib
from unittest import TestCase, skip
import pickle

from neat.link_prediction import model
from neat.link_prediction.sklearn_model import SklearnModel
from neat.yaml_helper.yaml_helper import YamlHelper

from sklearn.linear_model._logistic import LogisticRegression


class TestLinkPrediction(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.yaml_file = "tests/resources/test_neat.yaml"
        cls.yhelp = YamlHelper(cls.yaml_file)
        cls.test_model_path = "tests/resources/test_link_prediction/"
        cls.outfile = ((cls.yhelp.classifiers())[0])["model"]["outfile"]
        cls.model_outfile = cls.outfile.replace(".h5", "_model.h5")

    def setUp(self) -> None:
        pass

    def assertIsFile(self, path):
        if not pathlib.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_sklearn_save(self) -> None:
        model_object = SklearnModel(
            (self.yhelp.classifiers())[0], self.test_model_path
        )

        model_object.save()
        self.assertIsFile(os.path.join(self.test_model_path, self.outfile))
        self.assertIsFile(
            os.path.join(self.test_model_path, self.model_outfile)
        )

    def test_sklearn_load(self) -> None:
        out_fn = self.outfile
        out_model_fn = self.model_outfile

        with open(os.path.join(self.test_model_path, out_fn), "rb") as mf:
            model_object_1 = pickle.load(mf)

        with open(
            os.path.join(self.test_model_path, out_model_fn), "rb"
        ) as mf:
            model_object_2 = pickle.load(mf)
        self.assertEqual(type(model_object_1), LogisticRegression)
        self.assertEqual(type(model_object_2), SklearnModel)
