import os
import click
from neat.link_prediction.sklearn_model import SklearnModel
from neat.link_prediction.mlp_model import MLPModel

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
        config: Specify the YAML file containing instructions of what ML tasks to perform

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
            if classifier['type'] == 'neural network':
                model = MLPModel(classifier, outdir=yhelp.outdir())
            elif classifier['type'] in \
                    ['Decision Tree', 'Logistic Regression', 'Random Forest']:
                model = SklearnModel(classifier, outdir=yhelp.outdir())
            else:
                raise NotImplemented()

            model.compile()
            train_data, validation_data = \
                model.make_link_prediction_data(yhelp.embedding_outfile(),
                                                yhelp.main_graph_args(),
                                                yhelp.pos_val_graph_args(),
                                                yhelp.neg_train_graph_args(),
                                                yhelp.neg_val_graph_args(),
                                                yhelp.edge_embedding_method())
            model.fit(train_data, validation_data)
            model.save()

    return None


