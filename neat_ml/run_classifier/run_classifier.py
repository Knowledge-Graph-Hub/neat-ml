"""Run classifier."""
import os
from itertools import combinations  # [READ DOCS]
from pathlib import Path
from typing import List, Union
from warnings import warn

import pandas as pd  # type: ignore
from grape import Graph

from neat_ml.link_prediction.grape_model import GrapeModel  # type: ignore
from neat_ml.link_prediction.sklearn_model import SklearnModel

OUTPUT_COL_NAMES = ["source_node", "destination_node"]


def gen_src_dst_pair(
    graph: Graph,
    ignore_existing_edges: bool = True,
):
    """Generate source-destination pair.

    :param graph: Graph.
    :param ignore_existing_edges: Ignore edges or not., defaults to True
    :yield: Source-destination pair generation.
    """
    # Get all node ids
    node_ids = graph.get_node_ids().tolist()
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
    model: object,
    node_types: List[List],
    cutoff: float,
    output_file: Union[str, Path],
    embeddings_file: Union[str, Path],
    edge_method: str,  # [Average etc.]
    ignore_existing_edges: bool = True,
    verbose: bool = True,
) -> None:
    """Perform link prediction over provided graph nodes.

    Args:
        graph (Graph): Ensmallen graph.
        model (Any): Trained model.
        node_types (list): List of lists of target
        'source' and 'destination' nodes.
        Only these types will be in output.
        cutoff (float): Cutoff point for filtering.
        output_file (str or Path): Results destination.
        embeddings_file (str or Path): Path to embeddings.
        edge_method (str): Method to use for calculating edge embeddings.
        ignore_existing_edges (bool): default True; do not output
        predictions for edges already in graph.
    """
    print(f"Reading embeddings from {embeddings_file}...")
    embeddings = pd.read_csv(embeddings_file, sep=",", header=None)

    embedding_node_names = list(embeddings[0])
    src_dst_list = []
    no_embed_list = []

    print("Generating potential edges...")
    for src, dst in gen_src_dst_pair(graph, ignore_existing_edges):

        src_name = graph.get_node_name_from_node_id(src)
        dst_name = graph.get_node_name_from_node_id(dst)

        # Check if this pair passes the node filter(s)
        # Note that nodes may have >1 type
        # so these are lists
        if node_types:
            src_types = graph.get_node_type_names_from_node_id(src)
            dst_types = graph.get_node_type_names_from_node_id(dst)

            # Test if any intersection between node_types and src/dst types
            if (
                len(list(set(src_types) & set(node_types[0]))) == 0
                or len(list(set(dst_types) & set(node_types[1]))) == 0
            ):
                continue

        # see if src and dst are actually in embedding.tsv
        for name in [src_name, dst_name]:
            if name not in embedding_node_names:
                if verbose:
                    warn(f"Can't find {name} in embeddings - skipping")
                no_embed_list.append((src_name, dst_name))
            else:
                src_dst_list.append((src_name, dst_name))

    if len(src_dst_list) == 0:
        warn(
            "Filter has excluded all edges or no edges found. "
            "Cannot apply classifier."
        )

    edge_embedding_for_predict = model.make_edge_embedding_for_predict(  # type: ignore # noqa E501
        embedding_file=embeddings_file,  # this should be the new embeddings
        edge_method=edge_method,
        source_destination_list=src_dst_list,
    )

    embed_df = pd.DataFrame(src_dst_list, columns=OUTPUT_COL_NAMES)
    # NOTE: A trained Sklearn model treats '0' and '1' labels as classes
    #  as opposed to a Tensorflow(MLP) model where 0 and 1 are booleans
    # to a class (binary).

    print("Running edge predictions...")
    if type(model) == SklearnModel:
        pred_probas = [
            y for x, y in model.predict_proba(edge_embedding_for_predict)
        ]
        pred_proba_df = pd.DataFrame(pred_probas, columns=["score"])
        full_embed_df = pd.concat([embed_df, pred_proba_df], axis=1)
    elif type(model) == GrapeModel:
        nodemap = graph.get_nodes_mapping()
        inodemap = {value: key for key, value in nodemap.items()}
        preds = \
            model.predict_proba(graph=graph,
                                return_predictions_dataframe=True)
                    
        preds = preds.rename(columns={'predictions': 'score'})
        preds['source_node'] = \
            preds['sources'].map(lambda sources: inodemap[sources])
        preds['destination_node'] = \
            preds['destinations'].map(lambda destinations: 
                inodemap[destinations])

        # Ignore existing edges (i.e., only provide new edges)
        if ignore_existing_edges:
            print("Filtering existing edges...")
            preds = \
                preds[preds[['source_node',
                'destination_node']].apply(tuple, 1).isin(src_dst_list)]
        
        full_embed_df = preds

    else:
        preds = model.predict(edge_embedding_for_predict)  # type: ignore
        embed_df["score"] = preds
        full_embed_df = embed_df

    if no_embed_list:
        no_embed_df = pd.DataFrame(no_embed_list, columns=OUTPUT_COL_NAMES)
        output_df = pd.concat([full_embed_df, no_embed_df], axis=1)
    else:
        output_df = full_embed_df

    if cutoff:
        filtered_output = output_df[output_df["score"] > cutoff]
        output_df = filtered_output

    output_df.sort_values(by="score", inplace=True, ascending=False)
    output_df.to_csv(output_file, sep="\t", index=None)

    if len(output_df) > 0:
        print(f"Wrote predictions to {output_file}.")
    else:
        print("No edge predictions found meeting parameters.")


# This may be moved if needed
def get_custom_model_path(model_file_path: str) -> str:
    """Return the name of a custom model for TF/sklearn.

    This allows a NEAT Model object to be
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
