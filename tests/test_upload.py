import os
from unittest import TestCase
from neat.yaml_helper.yaml_helper import YamlHelper


class TestUpload(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.bad_yaml = 'tests/resources/test_bad_upload_info.yaml'
        self.good_yaml = 'tests/resources/test_good_upload_info.yaml'

    # def test_good_upload(self):
    #     yh = YamlHelper(self.good_yaml)
    #     embed_kwargs = yhelp.make_tsne_args()
    #
    #
    #     self.assertTrue(os.path.exists(self.expected_tsne_file))
