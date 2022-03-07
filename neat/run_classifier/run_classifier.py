from pathlib import Path
from typing import List
from embiggen import EdgeTransformer # type: ignore
from ensmallen import Graph  # type: ignore
import pandas as pd  # type: ignore
from itertools import combinations, permutations, product  # [READ DOCS]
import numpy as np


def predict_links(
    graph: Graph,
    model: object,
    node_types: List[List],
    cutoff: float,
    output_file: Path,
    embeddings: pd.DataFrame,
    edge_method: str,  # [Average etc.]
    ignore_existing_edges: bool = True,
) -> None:
    """Performs link prediction over provided graph nodes.

    Args:
        graph (Graph): Ensmallen graph.
        model (Any):Trained model.
        node_types (dict): Dictionary of 'source' and 'destination' nodes.
        cutoff (float): Cutoff point for filtering.
        output_file (Path): Results destination.
    """
    source_node_ids = [i for i, nt in enumerate(graph.get_node_type_names())
                       if any(x in nt for x in node_types[0])]
    destination_node_ids = [i for i, nt in enumerate(graph.get_node_type_names())
                            if any(x in nt for x in node_types[0])]

    with open(output_file, 'w') as f:
        for src in source_node_ids:
            for dst in destination_node_ids:
                if (graph.has_edge_from_node_ids(src, dst) or
                    (graph.has_edge_from_node_ids(dst, src) and not graph.is_directed())) \
                        and ignore_existing_edges:
                    continue
                source_embed = np.array(embeddings.loc[embeddings[0] == src])
                destination_embed = np.array(embeddings.loc[embeddings[0] == dst])
                predict_edges = model.make_link_prediction_predict_data()
                p = model.predict_proba(predict_edges)
                f.write("\t".join([src, dst, p]))

