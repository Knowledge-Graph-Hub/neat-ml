"""CLI for neat-ml."""
import json
import os

import click
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from grape import Graph  # type: ignore
from tqdm import tqdm  # type: ignore

import neat_ml.yaml_helper.config_fields as cf
from neat_ml.graph_embedding.graph_embedding import make_node_embeddings
from neat_ml.link_prediction.grape_model import GrapeModel
from neat_ml.link_prediction.mlp_model import MLPModel
from neat_ml.link_prediction.sklearn_model import SklearnModel
from neat_ml.method_names import GRAPE_LP_CLASS_NAMES, LR_NAMES, NN_NAMES
from neat_ml.pre_run_checks.pre_run_checks import pre_run_checks
from neat_ml.run_classifier.run_classifier import predict_links
from neat_ml.update_yaml.update_yaml import do_update_yaml
from neat_ml.upload.upload import upload_dir_to_s3
from neat_ml.visualization.visualization import make_tsne
from neat_ml.yaml_helper.yaml_helper import YamlHelper


@click.group()
def cli():
    """Click."""
    pass


@cli.command()
@click.option(
    "--config",
    required=True,
    default="config.yaml",
    type=click.Path(exists=True),
)
def run(config: str) -> None:
    """Run a NEAT pipeline using the given YAML file [neat.yaml].

    :param config: Specify the YAML file containing \
    instructions of what ML tasks to perform.
    :raises RuntimeError: If failed pre-check run.
    :raises NotImplementedError: If non-implemented model provided.
    :return: None
    """
    yhelp = YamlHelper(config)

    # pre run checks for failing early
    if not pre_run_checks(yhelp=yhelp, check_s3_credentials=yhelp.do_upload()):
        raise RuntimeError("Failed pre_run_check")

    # generate embeddings if config has 'embeddings' block
    if yhelp.do_embeddings() and not os.path.exists(yhelp.embedding_outfile()):
        node_embedding_args = yhelp.make_node_embeddings_args()
        make_node_embeddings(**node_embedding_args)

    if yhelp.do_tsne() and not os.path.exists(yhelp.tsne_outfile()):
        graph: Graph = yhelp.load_graph()
        tsne_kwargs = yhelp.make_tsne_args(graph)
        make_tsne(**tsne_kwargs)

    if yhelp.do_classifier():
        for classifier in tqdm(yhelp.classifiers()):

            # Check if classifier already exists
            if os.path.exists(yhelp.classifier_outfile(classifier)):
                classifier_id = classifier[cf.CLASSIFIER_ID]
                print(f"Found existing classifier: {classifier_id}")
                continue

            model: object = None
            if classifier[cf.CLASSIFIER_NAME].lower() in NN_NAMES:
                model = MLPModel(classifier, outdir=yhelp.outdir())
            elif (
                classifier[cf.CLASSIFIER_NAME].lower() in LR_NAMES
            ):
                model = SklearnModel(classifier, outdir=yhelp.outdir())
            elif (
                classifier[cf.CLASSIFIER_NAME].lower() in GRAPE_LP_CLASS_NAMES
            ):
                model = GrapeModel(classifier, outdir=yhelp.outdir())
            else:
                raise NotImplementedError(f"{model} isn't implemented yet")

            model.compile()

            train_data, validation_data = model.make_train_valid_data(
                embedding_file=yhelp.embedding_outfile(),
                training_graph_args=yhelp.main_graph_args(),
                validation_args=yhelp.val_graph_args(),
                training_args=yhelp.train_graph_args(),
                edge_method=yhelp.get_edge_embedding_method(classifier),
            )

            if type(model) in [SklearnModel, MLPModel]:
                history_obj = model.fit(*train_data)
            elif type(model) == GrapeModel:
                graph_obj = Graph.from_csv(**(yhelp.main_graph_args()))
                embed_obj = pd.read_csv(
                    (yhelp.embedding_outfile()), index_col=0, header=None
                )
                history_obj = model.fit(  # type: ignore
                    graph=graph_obj, node_features=embed_obj  # type: ignore
                )

            if type(model) == GrapeModel:
                if cf.POS_EDGE_FILEPATH in yhelp.val_graph_args():
                    val_graph_obj = Graph.from_csv(
                        node_path=yhelp.main_graph_args()[cf.NODE_PATH],
                        edge_path=yhelp.val_graph_args()[cf.POS_EDGE_FILEPATH],
                        nodes_column=yhelp.main_graph_args()[cf.NODES_COLUMN],
                        node_list_node_types_column=yhelp.main_graph_args()[
                            cf.NODE_LIST_NODE_TYPES_COLUMN
                        ],
                        directed=graph_obj.is_directed(),
                    )
                else:
                    val_graph_obj = graph_obj
                predicted = model.predict(graph=val_graph_obj)
                predicted_labels = (predicted * 1).tolist()
            elif type(model) == SklearnModel:
                predicted_labels = model.predict(validation_data[0])
            else:
                predicted_labels = np.concatenate(
                    np.around(model.predict(validation_data[0]), decimals=0)
                )
            actual_labels = validation_data[1].tolist()
            correct_matches = sum(
                [
                    1 if i == j else 0
                    for i, j in zip(predicted_labels, actual_labels)
                ]
            )
            total_data_points = len(validation_data[0])
            correct_label_match = (correct_matches / total_data_points) * 100

            print(f"Correct label match in validation: {correct_label_match}")

            if yhelp.classifier_history_file_name(classifier):
                with open(
                    yhelp.classifier_history_file_name(classifier), "w"
                ) as f:  # noqa E501
                    json.dump(history_obj.history, f)

            model.save()

    if yhelp.do_apply_classifier():
        # take graph, classifier, biolink node types and cutoff
        for clsfr_id in yhelp.get_classifier_id_for_prediction():
            classifier = yhelp.get_classifier_from_id(clsfr_id)
            if classifier[cf.CLASSIFIER_NAME].lower in GRAPE_LP_CLASS_NAMES:
                classifier_kwargs = yhelp.make_classifier_args(
                    clsfr_id, model  # type: ignore
                )
            else:
                classifier_kwargs = yhelp.make_classifier_args(clsfr_id)
            predict_links(**classifier_kwargs)

    if yhelp.do_upload():
        upload_kwargs = yhelp.make_upload_args()
        upload_dir_to_s3(**upload_kwargs)

    return None


@cli.command()
@click.option("--input_path", nargs=1, help="The path to the yaml to update.")
@click.option(
    "--keys",
    callback=lambda _, __, x: x.split(",") if x else [],
    help="One or more keys to update the values for, comma-delimited. "
    "Nested keys (i.e., keys under other keys) must be delimited "
    "with colons, e.g. key1:key2:key3.",
)
@click.option(
    "--values",
    callback=lambda _, __, x: x.split(",") if x else [],
    help="One or more values, in the same order as keys, comma-delimited.",
)
def updateyaml(input_path, keys, values) -> None:
    """Update a YAML file with specified key/value pairs.

    Updates one or more values for a one or more keys,
    with a provided path to a YAML file.
    Will not replace keys found multiple times in the YAML.
    Ignores keys in lists, even if they're dicts in lists.

    :param input_path: str, path to the yaml to edit
    :param keys: list of strs of keys to modify
    :param values: list of new values, in order of keys
    :return: None

    """
    do_update_yaml(input_path, keys, values)
