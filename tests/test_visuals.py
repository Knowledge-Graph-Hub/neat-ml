import os
from unittest import TestCase
from neat.yaml_helper.yaml_helper import YamlHelper
from neat.visualization.visualization import make_tsne, make_all_plots
from grape import Graph

class TestVisuals(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.expected_tsne_file = 'tests/resources/test_output_data_dir/tsne.png'
        self.expected_fullplot_file = 'tests/resources/test_output_data_dir/fullplots.png'
        for filepath in [self.expected_tsne_file, self.expected_fullplot_file]:
            if os.path.exists(filepath):
                print(
                    f"removing existing test tsne file {filepath}")
                os.unlink(filepath)

    def test_make_tsne(self):
        yhelp = YamlHelper("tests/resources/test.yaml")
        g = Graph.from_csv(
            nodes_column="id",
            node_list_node_types_column="category",
            default_node_type="biolink:NamedThing",
            node_path='tests/resources/test_graphs/test_small_nodes.tsv',
            edge_path='tests/resources/test_graphs/test_small_edges.tsv',
            sources_column="subject",
            destinations_column="object",
            directed=False
        )

        tsne_kwargs = yhelp.make_tsne_args(graph=g)
        tsne_kwargs['embedding_file'] = 'tests/resources/test_embeddings.tsv'
        make_tsne(**tsne_kwargs)

        self.assertTrue(os.path.exists(self.expected_tsne_file))

    def test_make_all_plots(self):
        yhelp = YamlHelper("tests/resources/test_for_plots.yaml")
        g = Graph.from_csv(
            nodes_column="id",
            node_list_node_types_column="category",
            default_node_type="biolink:NamedThing",
            node_path='tests/resources/test_graphs/test_small_nodes.tsv',
            edge_path='tests/resources/test_graphs/test_small_edges.tsv',
            sources_column="subject",
            destinations_column="object",
            directed=False
        )

        tsne_kwargs = yhelp.make_tsne_args(graph=g)
        tsne_kwargs['embedding_file'] = 'tests/resources/test_embeddings.tsv'
        make_all_plots(**tsne_kwargs)

        self.assertTrue(os.path.exists(self.expected_fullplot_file))
