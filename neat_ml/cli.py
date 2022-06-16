import json
import os
import click
from grape import Graph # type: ignore
import numpy as np  # type: ignore

from neat_ml.link_prediction.sklearn_model import SklearnModel
from neat_ml.link_prediction.mlp_model import MLPModel

from tqdm import tqdm  # type: ignore

from neat_ml.graph_embedding.graph_embedding import make_node_embeddings
from neat_ml.pre_run_checks.pre_run_checks import pre_run_checks
from neat_ml.update_yaml.update_yaml import do_update_yaml
from neat_ml.upload.upload import upload_dir_to_s3
from neat_ml.visualization.visualization import make_tsne
from neat_ml.run_classifier.run_classifier import predict_links
from neat_ml.yaml_helper.yaml_helper import YamlHelper


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    required=True,
    default="config.yaml",
    type=click.Path(exists=True),
)
def run(config: str) -> None:
    """Run a NEAT pipeline using the given YAML file [neat.yaml]
    \f

    Args:
        config: Specify the YAML file containing instructions of what ML tasks to perform

    Returns:
        None.

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
                classifier_id = classifier["classifier_id"]
                print(f"Found existing classifier: {classifier_id}")
                continue

            model: object = None
            if classifier["classifier_name"].lower() == "neural network":
                model = MLPModel(classifier, outdir=yhelp.outdir())
            elif classifier["classifier_name"].lower() in [
                "decision tree",
                "logistic regression",
                "random forest",
            ]:
                model = SklearnModel(classifier, outdir=yhelp.outdir())
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
            history_obj = model.fit(*train_data)
            if type(model) == SklearnModel:
                predicted_labels = model.predict(validation_data[0])
            else:
                predicted_labels = np.concatenate(
                    np.around(model.predict(validation_data[0]), decimals=0)
                )
            actual_labels = validation_data[1]
            correct_matches = sum(list(predicted_labels == actual_labels))
            total_data_points = len(validation_data[0])
            correct_label_match = (correct_matches / total_data_points) * 100

            print(f"Correct label match in validation: {correct_label_match}")

            if yhelp.classifier_history_file_name(classifier):
                with open(yhelp.classifier_history_file_name(classifier), "w") as f:  # type: ignore
                    json.dump(history_obj.history, f)

            model.save()

    if yhelp.do_apply_classifier():
        # take graph, classifier, biolink node types and cutoff
        for clsfr_id in yhelp.get_classifier_id_for_prediction():
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
def updateyaml(input_path, keys, values):
    """Update a YAML file with specified key/value pairs
    \f
    Updates one or more values for a one or more keys,
    with a provided path to a YAML file.
    Will not replace keys found multiple times in the YAML.
    Ignores keys in lists, even if they're dicts in lists.
    """
    do_update_yaml(input_path, keys, values)
