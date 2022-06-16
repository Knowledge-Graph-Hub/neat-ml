import pandas as pd  # type: ignore
from grape import Graph  # type: ignore
from grape import GraphVisualizer  # type: ignore


def make_tsne(
        graph: Graph,
        tsne_outfile: str,
        embedding_file: str,
) -> None:
    """
    Generate a simple node tSNE plot.
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

def make_all_plots(
        graph: Graph,
        tsne_outfile: str,
        embedding_file: str,
        ) -> None:
    """
    Generate the full set of Ensmallen plots.
    :param graph: Ensmallen graph object
    :param tsne_outfile: where to output plot image file
    :param embedding_file: file containing embeddings (in numpy format) to use to
        generate the tSNE plot
    :return:
    """

    node_embeddings = pd.read_csv(embedding_file, index_col=0, header=None)

    visualizer = GraphVisualizer(graph)
    figure = visualizer.fit_and_plot_all(node_embeddings)
    print(figure[1])
    figure[0].savefig(tsne_outfile)