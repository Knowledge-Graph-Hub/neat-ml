from unittest import TestCase
from neat.yaml_helper.yaml_helper import YamlHelper
from neat.visualization.visualization import make_tsne


class TestTsne(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def test_make_tsne(self):
        yhelp = YamlHelper("tests/resources/test_graph_embedding_bert_tsne.yaml")
        embed_kwargs = yhelp.make_tsne_args()
        make_tsne(**embed_kwargs)
