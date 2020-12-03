import os
import click

from neat.classifier import make_classifier, make_data, model_fit
from tqdm import tqdm

from neat.embeddings import make_tsne
from neat.graph_embedding.graph_embedding import make_embeddings
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
        kwargs = yhelp.make_embedding_args()
        make_embeddings(**kwargs)

    if 'tsne' in neat_config['embeddings']:
        make_tsne(neat_config)

    if 'classifier' in neat_config:
        for classifier in tqdm(neat_config['classifier']['classifiers']):
            model = make_classifier(classifier)
            train_data, validation_data = make_data(neat_config)
            model_fit(neat_config, model, train_data, validation_data, classifier)

    return None


