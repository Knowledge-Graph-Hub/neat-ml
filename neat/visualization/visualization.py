from typing import Any, Optional

import pandas as pd  # type: ignore
from MulticoreTSNE import MulticoreTSNE as TSNE  # type: ignore
from matplotlib import pyplot as plt  # type: ignore


def make_tsne(
        tsne_outfile: str,
        embedding_file: str,
        num_processors: int,
        scatter_params: dict,
        color_nodes: bool,
        node_file: str,
        node_property_for_color: str
) -> None:
    """

    :param tsne_outfile: where to output tSNE file
    :param embedding_file: file containing embeddings (in numpy format) to use to
        generate the tSNE plot
    :param num_processors: how many processors should we use
    :param scatter_params: parameters to pass to matplotlib scatter() method
    :param color_nodes: should we color nodes in tSNE plot?
    :param node_file: file containing node info for graph (used to retrieve node
        categories used to for colors in the tSNE plot (must be
        specified if color_nodes is specified)
    :param node_property_for_color: column in node file to use for colors (must be
        specified if color_nodes is specified)
    :return:
    """
    # fail early here while debugging:
    category_names: Optional[Any] = None
    colors: Optional[Any] = None
    cmap: Optional[Any] = None
    ticks: Optional[Any] = None
    if color_nodes:
        nodes = pd.read_csv(node_file, sep='\t')
        categories = nodes[node_property_for_color].astype(str)
        category_names = list(set(categories))
        category_names.sort()
        colors = [category_names.index(i) for i in categories]
        cmap = plt.cm.get_cmap('jet', len(category_names))
        ticks = list(range(len(category_names)))

    node_embeddings = pd.read_csv(embedding_file, index_col=0, header=None)
    tsne_embeddings = TSNE(n_jobs=num_processors).fit_transform(node_embeddings)
    x = tsne_embeddings[:, 0]
    y = tsne_embeddings[:, 1]
    plt.scatter(x, y, c=colors, cmap=cmap, **scatter_params)
    formatter = plt.FuncFormatter(lambda val, loc: category_names[val] if val in category_names else None)  # type: ignore
    plt.colorbar(ticks=ticks, format=formatter)
    plt.savefig(tsne_outfile)
