import os
from unittest import TestCase
from neat.yaml_helper.yaml_helper import YamlHelper
from neat.visualization.visualization import make_tsne
from ensmallen import Graph

class TestTsne(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.expected_tsne_file = 'output_data/test_embedding_tsne.png'
        if os.path.exists(self.expected_tsne_file):
            print(
                f"removing existing test tsne file {self.expected_tsne_file}")
            os.unlink(self.expected_tsne_file)

    def test_make_tsne(self):
        yhelp = YamlHelper("tests/resources/test_graph_embedding_bert_tsne.yaml")
        g = Graph.from_csv(
            nodes_column="id",
            node_list_node_types_column="category",
            default_node_type="biolink:NamedThing",
            node_path=yhelp.yaml['graph_data']['graph']['node_path'],
            edge_path=yhelp.yaml['graph_data']['graph']['edge_path'],
            sources_column="subject",
            destinations_column="object",
            directed=False
        )

        tsne_kwargs = yhelp.make_tsne_args(graph=g)
        tsne_kwargs['embedding_file'] = 'tests/resources/test_embeddings.tsv'
        make_tsne(**tsne_kwargs)

        self.assertTrue(os.path.exists(self.expected_tsne_file))
