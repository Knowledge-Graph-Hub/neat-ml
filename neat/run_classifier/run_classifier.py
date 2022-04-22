import os
from pathlib import Path
from typing import List, Union
from warnings import warn

from embiggen import EdgeTransformer  # type: ignore
from ensmallen import Graph  # type: ignore
import pandas as pd  # type: ignore
from itertools import combinations  # [READ DOCS]
import numpy as np

from neat.link_prediction.sklearn_model import SklearnModel

OUTPUT_COL_NAMES = ["source_node", "destination_node"]


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
    # training_graph_args: dict,
    # negative_graph_args: dict,
    model: object,
    node_types: List[List],
    cutoff: float,
    output_file: Union[str, Path],
    embeddings_file: Union[str, Path],
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
    embeddings = pd.read_csv(embeddings_file, sep=",", header=None)

    embedding_node_names = list(embeddings[0])
    src_dst_list = []
    no_embed_list = []

    for src, dst in gen_src_dst_pair(graph, ignore_existing_edges):

        src_name = graph.get_node_name_from_node_id(src)
        dst_name = graph.get_node_name_from_node_id(dst)

        # see if src and dst are actually in embedding.tsv:
        for name in [src_name, dst_name]:
            if not name in embedding_node_names:
                if verbose:
                    warn(f"Can't find {name} in embeddings - skipping")
                no_embed_list.append((src_name, dst_name))
            else:
                src_dst_list.append((src_name, dst_name))

    edge_embedding_for_predict = model.make_edge_embedding_for_predict(
        embedding_file=embeddings_file,  # this should be the new embeddings
        edge_method=edge_method,
        source_destination_list=src_dst_list,
    )

    embed_df = pd.DataFrame(src_dst_list, columns=OUTPUT_COL_NAMES)
    # NOTE: A trained Sklearn model treats '0' and '1' labels as classes
    #  as opposed to a Tensorflow(MLP) model where 0 and 1 are booleans
    # to a class (binary).

    if type(model) == SklearnModel:
        preds = model.predict(edge_embedding_for_predict).astype(int)
        embed_df["edge_type"] = preds
        pred_probas = model.predict_proba(edge_embedding_for_predict)
        pred_proba_df = pd.DataFrame(pred_probas)
        full_embed_df = pd.concat([embed_df, pred_proba_df], axis=1)
    else:
        preds = model.predict(edge_embedding_for_predict)
        embed_df["probabilities"] = preds
        full_embed_df = embed_df

    if no_embed_list:
        no_embed_df = pd.DataFrame(no_embed_list, columns=OUTPUT_COL_NAMES)
        output_df = pd.concat([full_embed_df, no_embed_df], axis=1)
    else:
        output_df = full_embed_df

    output_df.to_csv(output_file, sep="\t", index=None)


# This may be moved if needed
def get_custom_model_path(model_file_path: str) -> str:
    """
    Given the path to a sklearn or TF model,
    returns the name of the corresponding custom
    model. This allows a NEAT Model object to be
    created so we may access its methods.
    :param model_file_path: str, path to generic model
    :return: str, path to custom model
    """

    custom_model_path = (
        os.path.splitext(model_file_path)[0]
        + "_custom"
        + os.path.splitext(model_file_path)[1]
    )
    return custom_model_path
