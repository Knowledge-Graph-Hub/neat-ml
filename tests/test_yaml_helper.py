from unittest import TestCase
from neat.yaml_helper.yaml_helper import YamlHelper


class TestYamlHelper(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.test_yaml = "tests/resources/test.yaml"
        self.yh = YamlHelper(self.test_yaml)

    def test_no_indir(self) -> None:
        yh = YamlHelper("tests/resources/test_no_indir.yaml")
        self.assertEqual("", yh.indir())

    def test_bad_indir(self) -> None:
        with self.assertRaises(FileNotFoundError) as context:
            YamlHelper("tests/resources/test_bad_indir.yaml").indir()

    def test_outdir(self) -> None:
        self.assertEqual("output_data", self.yh.outdir())

    def test_add_indir_to_graph_data(self):
        # emits error message to log, but continues:
        self.yh.add_indir_to_graph_data(graph_data={}, keys_to_add_indir=['not_a_key'])
