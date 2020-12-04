import copy
import numpy as np
from typing import Tuple

from embiggen import GraphTransformer
from ensmallen_graph import EnsmallenGraph


def make_link_prediction_data(
        embedding_file: str,
        training_graph_args: dict,
        pos_validation_args: dict,
        neg_training_args: dict,
        neg_validation_args: dict,
        edge_method: str
) -> Tuple[Tuple, Tuple]:
    """Prepare training and validation data for training link prediction classifers

    Args:
        embedding_file: path to embedding file for nodes in graph
        training_graph_args: EnsmallenGraph arguments to load training graph
        pos_validation_args: EnsmallenGraph arguments to load positive validation graph
        neg_training_args: EnsmallenGraph arguments to load negative training graph
        neg_validation_args: EnsmallenGraph arguments to load negative validation graph
        edge_method: edge embedding method to use (average, L1, L2, etc)
    Returns:
        A tuple of tuples

    """
    embedding = np.load(embedding_file)

    # load graphs
    graphs = {'pos_training': EnsmallenGraph.from_unsorted_csv(**training_graph_args)}

    for name, graph_args in [('pos_validation', pos_validation_args),
                            ('neg_training', neg_training_args),
                            ('neg_validation', neg_validation_args)]:
        these_params = copy.deepcopy(training_graph_args)
        these_params.update(graph_args)
        graphs[name] = EnsmallenGraph.from_unsorted_csv(**these_params)

    # create graph transformer object to convert graphs into edge embeddings
    transformer = GraphTransformer(edge_method)
    transformer.fit(embedding)  # pass node embeddings to be used to create edge embeddings
    train_edges = np.vstack([  # computing edge embeddings for training graph
        transformer.transform(graph)
        for graph in (graphs['pos_training'], graphs['neg_training'])
    ])
    valid_edges = np.vstack([ # computing edge embeddings for validation graph
        transformer.transform(graph)
        for graph in (graphs['pos_validation'], graphs['neg_validation'])
    ])
    train_labels = np.concatenate([ # make labels for training graph
        np.ones(graphs['pos_training'].get_edges_number()),
        np.zeros(graphs['neg_training'].get_edges_number())
    ])
    valid_labels = np.concatenate([ # make labels for validation graph
        np.ones(graphs['pos_validation'].get_edges_number()),
        np.zeros(graphs['neg_validation'].get_edges_number())
    ])
    train_indices = np.arange(0, train_labels.size)
    valid_indices = np.arange(0, valid_labels.size)
    np.random.shuffle(train_indices) # shuffle to prevent bias caused by ordering of edge labels
    np.random.shuffle(valid_indices) # ``   ``
    train_edges = train_edges[train_indices]
    train_labels = train_labels[train_indices]
    valid_edges = valid_edges[valid_indices]
    valid_labels = valid_labels[valid_indices]
    return (train_edges, train_labels), (valid_edges, valid_labels)
