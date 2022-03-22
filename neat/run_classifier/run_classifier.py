from pathlib import Path
from typing import List
from warnings import warn

from embiggen import EdgeTransformer  # type: ignore
from ensmallen import Graph  # type: ignore
import pandas as pd  # type: ignore
from itertools import combinations  # [READ DOCS]
import numpy as np


def gen_src_dst_pair(
    graph: Graph,
    ignore_existing_edges: bool = True,
):

    # Get all node ids
    node_ids = graph.get_node_ids().tolist()[:100]
    # Yield only the (src, dst) combo
    # that does NOT exist in the graph.
    for combo in list(combinations(node_ids, 2)):
        # * 4 cases
        # 1. Graph DIRECTED & IGNORE Existing edges
        # 2. Graph DIRECTED & INCLUDE Existing edges
        # 3. Graph UNDIRECTED & IGNORE Existing edges
        # 4. Graph UNDIRECTED & INCLUDE Existing edges

        # Graph DIRECTED
        if graph.is_directed():
            # IGNORE Existing edges: Neither combo exist.
            if ignore_existing_edges:
                if not graph.has_edge_from_node_ids(
                    *combo
                ) and not graph.has_edge_from_node_ids(
                    *tuple(reversed(combo))
                ):
                    yield combo
            # INCLUDE Existing edges: yield every combo
            else:
                yield combo

        # Graph UNDIRECTED
        else:
            # IGNORE Existing edges:
            if ignore_existing_edges:
                if not graph.has_edge_from_node_ids(
                    *combo
                ) or not graph.has_edge_from_node_ids(*tuple(reversed(combo))):
                    yield combo
            # INCLUDE Existing edges: yield every combo
            else:
                yield combo


def predict_links(
    graph: Graph,
    training_graph_args: dict,
    model: object,
    node_types: List[List],
    cutoff: float,
    output_file: Path,
    embeddings_file: Path,
    edge_method: str,  # [Average etc.]
    ignore_existing_edges: bool = True,
    verbose: bool = True,
) -> None:
    """Performs link prediction over provided graph nodes.

    Args:
        graph (Graph): Ensmallen graph.
        model (Any):Trained model.
        node_types (dict): Dictionary of 'source' and 'destination' nodes.
        cutoff (float): Cutoff point for filtering.
        output_file (Path): Results destination.
    """

    # source_node_ids = [
    #     i
    #     for i, nt in enumerate(graph.get_node_type_names())
    #     if any(x in nt for x in node_types[0])
    # ]
    # destination_node_ids = [
    #     i
    #     for i, nt in enumerate(graph.get_node_type_names())
    #     if any(x in nt for x in node_types[0])
    # ]

    embeddings = pd.read_csv(embeddings_file, sep=",", header=None)

    #import pdb
    #pdb.set_trace()

    embedding_node_names = list(embeddings[0])
    with open(output_file, "w") as f:
        # for src in source_node_ids:
        #     for dst in destination_node_ids:
        #         if (
        #             graph.has_edge_from_node_ids(src, dst)
        #             or (
        #                 graph.has_edge_from_node_ids(dst, src)
        #                 and not graph.is_directed()
        #             )
        #         ) and ignore_existing_edges:
        #             continue
        for src, dst in gen_src_dst_pair(graph, ignore_existing_edges):

            src_name = graph.get_node_name_from_node_id(src)
            dst_name = graph.get_node_name_from_node_id(dst)

            have_embeddings = True

            # see if src and dst are actually in embedding.tsv:
            for name in [src_name, dst_name]:
                if not name in embedding_node_names:
                    if verbose:
                        warn(f"Can't find {name} in embeddings - skipping")
                    have_embeddings = False

            if not have_embeddings:
                f.write("\t".join([src_name, dst_name, "NaN\n"]))
                continue
            else:
                source_embed = np.array(
                    embeddings.loc[
                        embeddings[0] == graph.get_node_name_from_node_id(src)
                    ]
                )
                destination_embed = np.array(
                    embeddings.loc[
                        embeddings[0] == graph.get_node_name_from_node_id(dst)
                    ]
                )

                # TODO: create new embedding set based on source_embed and destination_embed
                #       then pass to make_link_prediction_predict_data (not in the loop)

        predict_edges = model.make_link_prediction_predict_data(embedding_file=embeddings_file,
                                                                trained_graph_args=training_graph_args,
                                                                edge_method=edge_method)
        p = model.predict_proba(predict_edges)
        f.write("\t".join([src, dst, p]) + "\n")

# This may be moved if needed
def get_custom_model(model_file_path: str) -> str:
    """
    Given the path to a sklearn or TF model,
    returns the name of the corresponding custom
    model. This allows a NEAT Model object to be
    created so we may access its methods.
    :param model_file_path: str, path to generic model
    :return: str, path to custom model
    """

    custom_model_path = (model_file_path.split("."))[0] + "_custom." + (model_file_path.split("."))[1] 

    return custom_model_path
