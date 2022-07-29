"""YAML helper."""
import functools
import logging
import os
import pickle
import string
import sys
import tarfile
from pathlib import Path
from typing import Optional, Union
from urllib.request import Request, urlopen

import pkg_resources  # type: ignore
import validators  # type: ignore
import yaml  # type: ignore
from grape import Graph  # type: ignore
from linkml_validator.validator import Validator  # type: ignore

from neat_ml.link_prediction.grape_model import GrapeModel
from neat_ml.run_classifier.run_classifier import \
     get_custom_model_path

VALID_CHARS = "-_.() %s%s" % (string.ascii_letters, string.digits)
INDIR_KEYS = ["node_path", "edge_path"]


def validate_config(
    config: dict, neat_schema_file: str = "neat_ml_schema.yaml"
) -> bool:
    """
    Validate the provided config against the neat_schema.

    :param config: dict of the parsed config file
    :param neat_schema_file: name of the neat schema
    file in neat-schema package
    :return: bool, false if validation failed
    """
    validated = True
    # Get schema path first
    schema_path = pkg_resources.resource_filename(
        "neat_ml_schema", os.path.join("schema/", neat_schema_file)
    )

    if not os.path.exists(schema_path):
        print(
            f"Cannot find {neat_schema_file}! \n"
            "Please verify that neat-ml-schema is installed and try again."
        )
        raise RuntimeError

    validator = Validator(schema=schema_path)
    for class_type in config:
        try:
            validator.validate(obj=config, target_class=class_type)
        except (KeyError, ValueError):
            validated = False
            print(f"Config failed validation for {class_type}")

    return validated


def parse_yaml(file: str) -> dict:
    """Parse YAML.

    :param file: YAML file path.
    :return: YAML file as a dict.
    """
    with open(file, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def is_url(string_to_check: Union[str, Path]) -> bool:
    """Decide if a string is a URL.

    (Used for example for deciding
    whether we need to download a file for a
    given node_path or edge_path).
    Raise exception if file path is invalid.

    :param string_to_check: string to check
    :return: bool, True if string is URL
    """
    return bool(validators.url(string_to_check))


def is_valid_path(string_to_check: Union[str, Path]) -> bool:
    """Validate file path.

    Raise exception if file path is invalid.

    :param string_to_check: string to check
    :return: bool, True if string is valid filepath
    """
    if isinstance(string_to_check, Path):
        if not string_to_check.is_file():
            raise FileNotFoundError(
                f"{string_to_check} is not a valid file path or url."
            )
    elif not os.path.exists(string_to_check):
        raise FileNotFoundError(
            f"{string_to_check} is not a valid file path or url."
        )
    else:
        return True

    return False


def download_file(url: str, outfile: str) -> list:
    """Download file at input url to outfile path.

    URL must point to a TSV or a tar.gz compressed file.
    (This is checked during pre_run_checks though.)
    If it's tar.gz, decompress.
    Return the names of all files as a list.
    """
    outlist = []

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response, open(outfile, "wb") as fh:  # type: ignore
        data = response.read()  # a `bytes` object
        fh.write(data)
    if outfile.lower().endswith(".tar.gz"):  # Need to decompress
        decomp_outfile = tarfile.open(outfile)
        outdir = os.path.dirname(outfile)
        for filename in decomp_outfile.getnames():
            outlist.append(os.path.join(outdir, filename))
        filecount = len(outlist)
        if filecount > 2:
            logging.warning(f"{outfile} contains {filecount} files.")
        decomp_outfile.extractall(outdir)
        decomp_outfile.close()
    else:
        outlist.append(outfile)

    return outlist


def catch_keyerror(f):
    """Neatly catch errors in missing YAML values."""

    @functools.wraps(f)
    def func(*args, **kwargs):
        """Check errors in missing YAML values and warn if missing."""
        try:
            return f(*args, **kwargs)
        except KeyError as e:
            print(f"Can't find key in YAML: {e} (possibly harmless)")
            return None

    return func


class YamlHelper:
    """Class to parse yaml and extract args for methods."""

    def __init__(self, config: str):
        """Initialize using configuration."""
        self.default_outdir = "output_data"
        self.default_indir = ""
        self.yaml: dict = parse_yaml(config)

        if not validate_config(self.yaml):
            sys.exit("Please check config file! Exiting...")

    def indir(self):
        """Get input directory.

        This is not currently specified in the
        config and will be the current
        working directory unless changed
        directly through a call to the
        YamlHelper object.

        Returns:
            The input directory

        """
        indir = self.default_indir

        return indir

    def outdir(self):
        """Get output directory from config.

        Returns:
            The output directory

        """
        outdir = (
            self.yaml["Target"]["target_path"]
            if "Target" in self.yaml
            else self.default_outdir
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return outdir

    def retrieve_from_sources(self) -> None:
        """Check for existence of a source_data key.

        If this exists,
        download and decompress as needed.
        The node_path and edge_path values
        in graph_data will still need to
        refer to the node/edge filenames.
        """
        if "GraphDataConfiguration" in self.yaml:
            if "source_data" in self.yaml["GraphDataConfiguration"]:
                for entry in self.yaml["GraphDataConfiguration"][
                    "source_data"
                ]["files"]:
                    filepath = entry["path"]
                    if "desc" in entry:
                        desc = entry["desc"]
                        print(f"Retrieving {filepath}: {desc}")
                    else:
                        print(f"Retrieving {filepath}")
                    if is_url(filepath):
                        url_as_filename = "".join(
                            c if c in VALID_CHARS else "_" for c in filepath
                        )
                        outfile = os.path.join(self.indir(), url_as_filename)
                        download_file(filepath, outfile)
                    # If this was a URL, it already got decompressed.
                    # but if it's local and still compressed, decompress now
                    # (this can happen if we already downloaded it
                    # but didn't decomp)
                    if filepath.endswith(".tar.gz"):
                        outlist = []
                        if is_url(filepath):
                            decomp_outfile = tarfile.open(outfile)
                        else:
                            decomp_outfile = tarfile.open(filepath)
                        for filename in decomp_outfile.getnames():
                            outlist.append(
                                os.path.join(self.indir(), filename)
                            )
                        decomp_outfile.extractall(self.indir())
                        decomp_outfile.close()

    def add_indir_to_graph_data(
        self,
        graph_data: dict,
        keys_to_add_indir: list = INDIR_KEYS,
    ) -> dict:
        """Update the graph file paths with input directory.

        :param graph_data - parsed yaml
        :param keys_to_add_indir: what keys to add indir to
        :return:
        """
        for k in keys_to_add_indir:
            if k in graph_data:
                graph_data[k] = os.path.join(self.indir(), graph_data[k])
            else:
                logging.warning(
                    f"Can't find key {k} in graph_data - skipping\
                        (possibly harmless)"
                )
        return graph_data

    #
    # graph stuff
    #
    def load_graph(self) -> Graph:
        """Load graph nodes and edges into Ensmallen.

        Create a node type list, as Ensmallen
        requires this to parse node types.
        :return: ensmallen Graph
        """
        # Load sources if necessary
        self.retrieve_from_sources()

        graph_args_with_indir = self.main_graph_args()

        for pathtype in ["node_path", "edge_path"]:
            filepath = graph_args_with_indir[pathtype]
            if is_url(filepath):
                url_as_filename = "".join(
                    c if c in VALID_CHARS else "_" for c in filepath
                )
                outfile = os.path.join(self.outdir(), url_as_filename)
                download_file(filepath, outfile)
                graph_args_with_indir[pathtype] = os.path.join(outfile)
            elif not is_valid_path(filepath):
                raise FileNotFoundError(f"Please check path: {filepath}")

        # Now load the Ensmallen graph
        loaded_graph = Graph.from_csv(**graph_args_with_indir)

        return loaded_graph

    def main_graph_args(self) -> dict:
        """Graph arguments."""
        return self.add_indir_to_graph_data(
            self.yaml["GraphDataConfiguration"]["graph"]
        )

    @catch_keyerror
    def val_graph_args(self) -> dict:
        """Assemble dictionary of validation graph parameters."""
        return self.add_indir_to_graph_data(
            self.yaml["GraphDataConfiguration"]["evaluation_data"][
                "valid_data"
            ]
        )

    @catch_keyerror
    def train_graph_args(self) -> dict:
        """Assemble dictionary of training graph parameters."""
        return self.add_indir_to_graph_data(
            self.yaml["GraphDataConfiguration"]["evaluation_data"][
                "train_data"
            ]
        )

    #
    # embedding stuff
    #

    def do_embeddings(self) -> bool:
        """Check if the config includes embedding generation."""
        return "EmbeddingsConfig" in self.yaml

    def embedding_outfile(self) -> str:
        """Return full path to embedding file."""
        filepath = self.yaml["EmbeddingsConfig"]["filename"]
        if is_url(filepath):
            url_as_filename = "".join(
                c if c in VALID_CHARS else "_" for c in filepath
            )
            outfile = os.path.join(self.outdir(), url_as_filename)
            download_file(filepath, outfile)
            return outfile
        else:
            return os.path.join(self.outdir(), filepath)

    @catch_keyerror
    def embedding_history_outfile(self):
        """Return full path to embedding history file."""
        return os.path.join(
            self.outdir(),
            self.yaml["EmbeddingsConfig"]["history_filename"],
        )

    # def make_embeddings_metrics_class_list(self) -> list:
    #     metrics_class_list = []

    #     metrics = (
    #         self.yaml["EmbeddingsConfig"]["metrics"]
    #         if "metrics" in self.yaml["embeddings"]
    #         else None
    #     )
    #     if metrics:
    #         for m in metrics:
    #             if m["type"].startswith("tensorflow.keras"):
    #                 m_class = Model.dynamically_import_class(m["type"])
    #                 m_parameters = m["parameters"]
    #                 m_instance = m_class(**m_parameters)  # type: ignore
    #                 metrics_class_list.append(m_instance)
    #             else:
    #                 metrics_class_list.append([m["type"]])
    #     return metrics_class_list

    def make_node_embeddings_args(self) -> dict:
        """Prepare a dict of parameters for node embeddings."""
        node_embedding_args = {
            "embedding_outfile": self.embedding_outfile(),
            "embedding_history_outfile": self.embedding_history_outfile(),
            "main_graph_args": self.main_graph_args(),
            "node_embedding_params": self.yaml["EmbeddingsConfig"][
                "node_embeddings_params"
            ],
            "bert_columns": self.yaml["EmbeddingsConfig"]["bert_params"][
                "node_columns"
            ]
            if "bert_params" in self.yaml["EmbeddingsConfig"]
            else None,
        }
        return node_embedding_args

    #
    # tSNE stuff
    #

    def do_tsne(self) -> bool:
        """Check if the config includes tSNE plotting."""
        return "tsne_filename" in self.yaml["EmbeddingsConfig"]

    def make_tsne_args(self, graph: Graph) -> dict:
        """Assemble provided parametes for tSNE plotting."""
        make_tsne_args = {
            "graph": graph,
            "tsne_outfile": self.tsne_outfile(),
            "embedding_file": self.embedding_outfile(),
        }
        return make_tsne_args

    def tsne_outfile(self) -> str:
        """Return full path to tSNE plot."""
        return os.path.join(
            self.outdir(), self.yaml["EmbeddingsConfig"]["tsne_filename"]
        )

    #
    # classifier stuff
    #

    def do_classifier(self) -> bool:
        """Check if the config includes classifiers."""
        return "ClassifierContainer" in self.yaml

    def classifier_type(self) -> str:
        """Return the type (i.e., name) of classifier."""
        return self.yaml["ClassifierContainer"]["classifiers"][
            "classifier_name"
        ]

    @catch_keyerror
    def classifiers(self) -> list:
        """From the YAML, extract a list of classifiers to be trained.

        :return: list of classifiers to be trained
        """
        return self.yaml["ClassifierContainer"]["classifiers"]

    def get_all_classifier_ids(self) -> list:
        """Return list of classifier ids."""
        return [
            c["classifier_id"]
            for c in self.yaml["ClassifierContainer"]["classifiers"]
        ]

    def get_edge_embedding_method(self, classifier: dict) -> str:
        """Return value for edge method for classifier."""
        return classifier["edge_method"]

    def classifier_history_file_name(self, classifier: dict) -> str:
        """Return full path to classifier history file."""
        return (
            classifier["history_filename"]
            if "history_filename" in classifier
            else None
        )

    def classifier_outfile(self, classifier: dict) -> str:
        """Return full path to classifier file."""
        return os.path.join(self.outdir(), classifier["outfile"])

    #
    # upload stuff
    #

    def do_upload(self) -> bool:
        """Upload."""
        return "Upload" in self.yaml

    def make_upload_args(self) -> dict:
        """Get upload arguments.

        :return: Dictionary of upload arguments.
        """
        make_upload_args = {
            "local_directory": self.outdir(),
            "s3_bucket": self.yaml["Upload"]["s3_bucket"],
            "s3_bucket_dir": self.yaml["Upload"]["s3_bucket_dir"],
            "extra_args": self.yaml["Upload"]["extra_args"]
            if "extra_args" in self.yaml["Upload"]
            else None,
        }
        return make_upload_args

    #
    # applying trained model to fresh data for predictions
    #
    def do_apply_classifier(self) -> bool:
        """Check on whether to apply classifier.

        :return: bool, True if should apply
        """
        return "ApplyTrainedModelsContainer" in self.yaml

    def get_classifier_id_for_prediction(self):
        """Get classifier ID for prediction.

        :return: List of classifier IDs.
        """
        classifier_applications = self.yaml["ApplyTrainedModelsContainer"][
            "models"
        ]
        list_of_ids = [cl["model_id"] for cl in classifier_applications]
        return list_of_ids

    def get_classifier_from_id(self, classifier_id: str):
        """Get classifier from ID.

        :param classifier_id: Classifier ID.
        :return: Classifier information.
        """
        return [
            x
            for x in self.classifiers()
            if x["classifier_id"] == classifier_id
        ][0]

    def make_classifier_args(
        self, cl_id: str, model_in: Optional[GrapeModel] = None
    ) -> dict:
        """Make classifier arguments.

        :param cl_id: Classifier ID.
        :param model_in: a pre-built model, if available
        :return: Classifier argument dictionary.
        """
        classifier_args = [
            arg
            for arg in self.yaml["ApplyTrainedModelsContainer"]["models"]
            if cl_id == arg["model_id"]
        ][0]
        model = self.get_classifier_from_id(cl_id)

        classifier_args_dict = {}
        classifier_args_dict["graph"] = Graph.from_csv(
            **self.main_graph_args()
        )

        if (
            self.get_classifier_from_id(cl_id)["classifier_name"]
            == "neural network"
        ):
            classifier_args_dict["model"] = pickle.load(
                open(
                    os.path.join(
                        self.outdir(),
                        get_custom_model_path(model["outfile"]),
                    ),
                    "rb",
                ),
            )

        elif model_in is not None:
            # Workaround for grape save/load not implemented yet
            # We just take the model object as an argument
            classifier_args_dict["model"] = model_in
        else:
            classifier_args_dict["model"] = pickle.load(
                open(
                    os.path.join(self.outdir(), model["outfile"]),
                    "rb",
                ),
            )

        # YAML may specify node_types as dict with 'source' and 'destination'
        # or as a list of lists
        if (
            "source" in classifier_args["node_types"]
            or "destination" in classifier_args["node_types"]
        ):
            classifier_args_dict["node_types"] = [
                classifier_args["node_types"]["source"],
                classifier_args["node_types"]["destination"],
            ]
        else:
            classifier_args_dict["node_types"] = classifier_args["node_types"]

        classifier_args_dict["cutoff"] = classifier_args["cutoff"]
        classifier_args_dict["output_file"] = os.path.join(
            self.outdir(), classifier_args["outfile"]
        )

        classifier_args_dict["embeddings_file"] = self.embedding_outfile()
        classifier_args_dict["edge_method"] = model["edge_method"]

        return classifier_args_dict
