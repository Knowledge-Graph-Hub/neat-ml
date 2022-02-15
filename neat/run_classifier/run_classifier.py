from pathlib import Path
from typing import Any
from ensmallen import Graph


def predict_links(
    graph: Graph,
    model: Any,
    node_types: dict,
    cutoff: float,
    output_file: Path,
):
    """Performs link prediction over provided graph nodes.

    Args:
        graph (Graph): Ensmallen graph.
        model (Any):Trained model.
        node_types (dict): Dictionary of 'source' and 'destination' nodes.
        cutoff (float): Cutoff point for filtering.
        output_file (Path): Results destination.
    """
    pass
