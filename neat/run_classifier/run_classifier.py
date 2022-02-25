from pathlib import Path
from xmlrpc.client import boolean
from embiggen import EdgeTransformer
from ensmallen import Graph  # type: ignore
import pandas as pd
from itertools import combinations, permutations, product  # [READ DOCS]
import numpy as np
from scipy.fft import dst
from sympy.utilities.iterables import multiset_permutations


def predict_links(
    graph: Graph,
    model: object,
    node_types: dict,
    cutoff: float,
    output_file: Path,
    embeddings: pd.DataFrame,
    edge_method: str,  # [Average etc.]
    ignore_existing_edges: bool = True,
):
    """Performs link prediction over provided graph nodes.

    Args:
        graph (Graph): Ensmallen graph.
        model (Any):Trained model.
        node_types (dict): Dictionary of 'source' and 'destination' nodes.
        cutoff (float): Cutoff point for filtering.
        output_file (Path): Results destination.
    """

    # generate every possible combo of nodes (source => dest) [need Graph object]
    #   a. get_node_from .... => 2 lists => src_type and dest_type
    #       -> graph.get_nodes_mappings()
    #   b. combinations_with_replacement will give every combo # [READ DOCS]
    #       - IF does not already exist in graph
    #           graph.get_edge_node_ids(directed=False)
    #           graph.get_edge_node_names(directed=False)
    #   c. For each combo, retrieve src and dest embeddings & apply edge method => edge_embedding.
    #       Give 2 node embeddings and retrieve the corresponding edge embedding
    #   d. Feed edge_embedding => model == score
    #   e. if score >= cutoff => accept.
    #   Final O/p: "source_node" -> "edge" -> "dest_node" -> "score"[ONLY above & equal to cutoff]

    # TODO: Fix the next 2 lines once node_types bug is fixed in ensmallen
    # 1. Get source nodes for nodes of types: ['biolink:XXXX', 'biolink:XYXY']
    # 2. Get destination nodes for nodes of types: ['biolink:YYYY', 'biolink:YXYX']
    source_node_ids = graph.get_random_nodes(10, random_state=1)
    destination_node_ids = graph.get_random_nodes(10, random_state=10)

    non_existant_node_combo = [
        combo
        for combo in product(source_node_ids, destination_node_ids)
        if not graph.has_edge_from_node_ids(combo[0], combo[1])
    ]

    for combo in non_existant_node_combo:
        source_node = graph.get_node_name_from_node_id(combo[0])
        destination_node = graph.get_node_name_from_node_id(combo[1])
        source_embed = np.array(
            embeddings.loc[embeddings[0] == source_node]  # .iloc[:, 1:]
        )
        destination_embed = np.array(
            embeddings.loc[embeddings[0] == destination_node]  # .iloc[:, 1:]
        )
        # * Trying to get edge embeddings from source-dest node embeddings.
        # edge_transformer = EdgeTransformer()
        # edge_transformer.fit(embedding=embeddings)
        # edge_transformer.transform()

    # * ## Need to decide if this block is relevant or no.#####
    all_node_type_names = graph.get_node_type_names()
    unique_node_type_names = list(
        set([x[0] for x in all_node_type_names if x])
    )

    for k, v in node_types.items():
        if not all(unique_node_type_names) in v:
            print(v)
    # * ######################################################
