import logging
import os
from typing import List, Optional

from ensmallen_graph import EnsmallenGraph  # type: ignore


def make_holdouts(main_graph_args: dict, output_dir: str,
                  train_size: float, validation: bool,
                  edge_types: List[str] = None,
                  seed=42) -> None:
    """Prepare positive and negative edges for testing and training (see run.py holdouts
    command for documentation)

    Args:
        :param main_graph_args: how to load the main graph (produced by YamlHelper().main_graph_args()
        :param output_dir:      where should we write out holdout data
        :param train_size:      fraction of edges to emit as training
        :param edge_types:      what edge types should we select for positive test/validation edges?
        :param validation:      should we make validation edges? [False]
        :param seed:            random seed [42]
    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading graph")
    graph = EnsmallenGraph.from_unsorted_csv(**main_graph_args)

    # make positive edges
    logging.info("Making positive edges")
    pos_train_edges, pos_test_edges = graph.random_holdout(random_state=seed,
                                                           train_size=train_size,
                                                           edge_types=edge_types)
    pos_valid_edges: Optional[EnsmallenGraph] = None
    if validation:
        pos_valid_edges, pos_test_edges = \
            pos_test_edges.random_holdout(random_state=seed,
                                          train_size=0.5)

    # make negative edges
    logging.info("Making negative edges")

    all_negative_edges = \
        pos_train_edges.sample_negatives(random_state=seed,
                                         negatives_number=graph.get_edges_number())
    neg_train_edges, neg_test_edges = \
        all_negative_edges.random_holdout(random_state=seed, train_size=train_size)
    neg_valid_edges = None
    if validation:
        neg_test_edges, neg_valid_edges = \
            neg_test_edges.random_holdout(random_state=seed, train_size=0.5)

    #
    # write out positive edges
    #
    # training:
    logging.info("Writing out positive edges")
    pos_train_edges_outfile = os.path.join(output_dir, "pos_train_edges.tsv")
    pos_train_nodes_outfile = os.path.join(output_dir, "pos_train_nodes.tsv")
    pos_test_edges_outfile = os.path.join(output_dir, "pos_test_edges.tsv")
    pos_valid_edges_outfile = os.path.join(output_dir, "pos_valid_edges.tsv")

    pos_train_edges.dump_edges(path=pos_train_edges_outfile)
    pos_train_edges.dump_nodes(path=pos_train_nodes_outfile)
    pos_test_edges.dump_edges(path=pos_test_edges_outfile)
    if validation:
        pos_valid_edges.dump_edges(path=pos_valid_edges_outfile)  # type: ignore

    #
    # write out negative edges
    #
    logging.info("Writing out negative edges")
    neg_train_edges_outfile = os.path.join(output_dir, "neg_train_edges.tsv")
    neg_test_edges_outfile = os.path.join(output_dir, "neg_test_edges.tsv")
    neg_valid_edges_outfile = os.path.join(output_dir, "neg_valid_edges.tsv")

    neg_train_edges.dump_edges(path=neg_train_edges_outfile)
    neg_test_edges.dump_edges(path=neg_test_edges_outfile)
    if validation:
        neg_valid_edges.dump_edges(path=neg_valid_edges_outfile)  # type: ignore
