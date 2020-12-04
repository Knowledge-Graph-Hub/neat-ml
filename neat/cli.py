import os
import click

from neat.classifier import make_classifier, make_data, model_fit
from tqdm import tqdm

from neat.graph_embedding.graph_embedding import make_embeddings
from neat.visualization.visualization import make_tsne
from neat.yaml_helper.yaml_helper import YamlHelper


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", required=True, default="config.yaml", type=click.Path(exists=True))
def run(config: str) -> None:
    """Run a NEAT pipeline using the given YAML file [neat.yaml]
    \f

    Args:
        config: Specify the YAML file containing a list of datasets to download.

    Returns:
        None.

    """

    yhelp = YamlHelper(config)

    # generate embeddings if config has 'embeddings' block
    if yhelp.do_embeddings() and not os.path.exists(yhelp.embedding_outfile()):
        embed_kwargs = yhelp.make_embedding_args()
        make_embeddings(**embed_kwargs)

    if yhelp.do_tsne() and not os.path.exists(yhelp.tsne_outfile()):
        tsne_kwargs = yhelp.make_tsne_args()
        make_tsne(**tsne_kwargs)

    if yhelp.do_classifier():
        for classifier in tqdm(yhelp.classifiers()):
            model = make_classifier(classifier)
            train_data, validation_data = make_data(neat_config)
            model_fit(neat_config, model, train_data, validation_data, classifier)

    return None


