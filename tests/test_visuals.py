import os
from unittest import TestCase
from neat_ml.visualization.visualization import make_tsne, make_all_plots
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
        tsne_kwargs = {"graph": g,
            "tsne_outfile": self.expected_tsne_file,
            "embedding_file": 'tests/resources/test_embeddings.csv',
            }
        make_all_plots(**tsne_kwargs)

        self.assertTrue(os.path.exists(self.expected_tsne_file))

    def test_make_all_plots(self):

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
        tsne_kwargs = {"graph": g,
            "tsne_outfile": self.expected_fullplot_file,
            "embedding_file": 'tests/resources/test_embeddings.csv',
            }
        make_all_plots(**tsne_kwargs)

        self.assertTrue(os.path.exists(self.expected_fullplot_file))
