"""Run classifier."""
import os
from pathlib import Path
from typing import List, Union
from warnings import warn

import pandas as pd  # type: ignore
from grape import Graph

from neat_ml.link_prediction.grape_model import GrapeModel  # type: ignore

OUTPUT_COL_NAMES = ["source_node", "destination_node"]


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
    no_embed_list = []

    print("Generating potential edges...")
    candidate_graph = graph.sample_negative_graph(
        number_of_negative_samples=graph.get_number_of_edges()
    )

    # Filter nodes by node type
    if node_types:
        candidate_graph = candidate_graph.filter_from_names(
            node_type_name_to_keep=node_types
        )

    # Remove nodes if they aren't in the provided embedding
    candidate_graph = candidate_graph.filter_from_names(
        node_names_to_keep=embedding_node_names
    )

    if (
        candidate_graph.get_nodes_number() == 0
        or candidate_graph.get_edges_number() == 0
    ):
        warn(
            "Filter has excluded all edges or no edges found. "
            "Cannot apply classifier."
        )

    # NOTE: A trained Sklearn model treats '0' and '1' labels as classes
    #  as opposed to a Tensorflow(MLP) model where 0 and 1 are booleans
    # to a class (binary).

    print("Running edge predictions...")
    if type(model) == GrapeModel:
        preds = model.predict_proba(
            graph=graph, return_predictions_dataframe=True
        )

        preds = preds.rename(columns={"predictions": "score"})

        # Ignore existing edges (i.e., only provide new edges)

        if ignore_existing_edges:
            cols = ["sources", "destinations"]
            all_edge_node_names = graph.get_edge_node_names(
                directed=graph.is_directed()
            )
            preds = preds[
                ~preds[cols].apply(tuple, 1).isin(all_edge_node_names)
            ]

        # Remove any self-interactions
        preds = preds[preds["sources"] != preds["destinations"]]

        full_embed_df = preds

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
