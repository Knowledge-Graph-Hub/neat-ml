import os
from unittest import TestCase, skip
import pickle

from neat.link_prediction import model
from neat.link_prediction.sklearn_model import SklearnModel
from neat.yaml_helper.yaml_helper import YamlHelper

class TestLinkPrediction(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.yaml_file = "tests/resources/test_neat.yaml"
        cls.yhelp = YamlHelper(cls.yaml_file)
        cls.test_model_path = "tests/resources/test_link_prediction/"

    def setUp(self) -> None:
        pass

    def test_sklearn_save(self) -> None:
        model_object = SklearnModel((self.yhelp.classifiers())[0], 
                                    self.test_model_path)

        model_object.save()

    def test_sklearn_load(self) -> None:
        out_fn = ((self.yhelp.classifiers())[0])['model']['outfile']
        import pdb
        pdb.set_trace()
        model_object = pickle.load(os.path.join(self.test_model_path, out_fn))


