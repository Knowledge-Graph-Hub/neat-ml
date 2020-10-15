import os

from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence
import tempfile


def make_embeddings(config: dict) -> None:
    graph = EnsmallenGraph.from_unsorted_csv(**config['graph'])
    graph_sequence = Node2VecSequence(graph, **config['embiggen_params']['node2vec_params'])
    return None


def make_classifier(config: dict) -> None:
    pass
