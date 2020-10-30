import os
import click
import yaml

from neat.embeddings import make_embeddings, get_output_dir, make_tsne
from neat.classifier import make_classifier, make_data, model_fit
from tqdm import tqdm


def parse_yaml(file: str) -> object:
    with open(file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


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
    neat_config = parse_yaml(config)

    # generate embeddings if config has 'embeddings' block
    if 'embeddings' in neat_config:
        if not os.path.exists(
                os.path.join(get_output_dir(neat_config),
                             neat_config['embeddings']['embedding_file_name'])):
            make_embeddings(neat_config)
            if 'tsne' in neat_config['embeddings']:
                make_tsne(neat_config)


    if 'classifier' in neat_config:
        for classifier in tqdm(neat_config['classifier']['classifiers']):
            model = make_classifier(classifier)
            train_data, validation_data = make_data(neat_config)
            model_fit(neat_config, model, train_data, validation_data, classifier)

    return None


