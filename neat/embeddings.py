import os

from ensmallen_graph import EnsmallenGraph
import tempfile


def make_embeddings(config: dict) -> None:
    sorted_edges = os.path.join(tempfile.mkdtemp(), 'sorted_edges.tsv')
    graph = EnsmallenGraph.from_unsorted_csv(**config['graph'])
    return None


def make_classifier(config: dict) -> None:
    pass
