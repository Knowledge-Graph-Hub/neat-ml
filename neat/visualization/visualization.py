import pandas as pd  # type: ignore
from grape import Graph  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from grape import GraphVisualizer  # type: ignore


def make_tsne(
        graph: Graph,
        tsne_outfile: str,
        embedding_file: str,
) -> None:
    """
    :param graph: Ensmallen graph object
    :param tsne_outfile: where to output tSNE file
    :param embedding_file: file containing embeddings (in numpy format) to use to
        generate the tSNE plot
    :return:
    """

    node_embeddings = pd.read_csv(embedding_file, index_col=0, header=None)

    visualizer = GraphVisualizer(graph)
    figure = visualizer.plot_nodes(visualizer.fit_nodes(node_embeddings))
    print(figure[1])
    figure[0].savefig(tsne_outfile)
