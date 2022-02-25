import functools
import logging
import os
import pickle
import string
from typing import Optional, Callable, Any
from urllib.request import Request, urlopen

import yaml  # type: ignore
from ensmallen import Graph  # type: ignore
from neat.link_prediction.model import Model
import validators  # type: ignore
import pandas as pd


def parse_yaml(file: str) -> dict:
    with open(file, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def is_url(string_to_check: str) -> bool:
    """Helper function to decide if a string is a URL (used for example for deciding
    whether we need to download a file for a given node_path or edge_path

    :param string_to_check: string to check
    :return: True/False is this a URL
    """
    return bool(validators.url(string_to_check))


def download_file(url: str, outfile: str) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response, open(outfile, "wb") as fh:  # type: ignore
        data = response.read()  # a `bytes` object
        fh.write(data)


def catch_keyerror(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyError as e:
            print("can't find key in YAML: ", e, "(possibly harmless)")
            return None

    return func


class YamlHelper:
    """
    Class to parse yaml and extract args for methods
    """

    def __init__(self, config: str):
        self.default_outdir = "output_data"
        self.default_indir = ""
        self.yaml: dict = parse_yaml(config)

    def indir(self):
        """Get input directory from config.

        Returns:
            The input directory

        """
        if "input_directory" in self.yaml:
            indir = self.yaml["input_directory"]
            if not os.path.exists(indir):
                raise FileNotFoundError(f"Can't find input dir {indir}")
        else:
            indir = self.default_indir
        return indir

    def outdir(self):
        """Get output directory from config.

        Returns:
            The output directory

        """
        outdir = (
            self.yaml["output_directory"]
            if "output_directory" in self.yaml
            else self.default_outdir
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return outdir

    def add_indir_to_graph_data(
        self,
        graph_data: dict,
        keys_to_add_indir: list = ["node_path", "edge_path"],
    ) -> dict:
        """
        :param graph_data - parsed yaml
        :param keys_to_add_indir: what keys to add indir to
        :return:
        """
        for k in keys_to_add_indir:
            if k in graph_data:
                graph_data[k] = os.path.join(self.indir(), graph_data[k])
            else:
                logging.warning(
                    f"Can't find key {k} in graph_data - skipping (possibly harmless)"
                )
        return graph_data

    #
    # graph stuff
    #
    def main_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(self.yaml["graph_data"]["graph"])

    @catch_keyerror
    def pos_val_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(
            self.yaml["graph_data"]["pos_validation"]
        )

    @catch_keyerror
    def neg_val_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(
            self.yaml["graph_data"]["neg_validation"]
        )

    @catch_keyerror
    def neg_train_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(
            self.yaml["graph_data"]["neg_training"]
        )

    #
    # embedding stuff
    #

    def do_embeddings(self) -> bool:
        return "embeddings" in self.yaml

    def embedding_outfile(self) -> str:
        return os.path.join(
            self.outdir(), self.yaml["embeddings"]["embedding_file_name"]
        )

    @catch_keyerror
    def embedding_history_outfile(self):
        return os.path.join(
            self.outdir(),
            self.yaml["embeddings"]["embedding_history_file_name"],
        )

    def make_embeddings_metrics_class_list(self) -> list:
        metrics_class_list = []

        metrics = (
            self.yaml["embeddings"]["metrics"]
            if "metrics" in self.yaml["embeddings"]
            else None
        )
        if metrics:
            for m in metrics:
                if m["type"].startswith("tensorflow.keras"):
                    m_class = Model.dynamically_import_class(m["type"])
                    m_parameters = m["parameters"]
                    m_instance = m_class(**m_parameters)  # type: ignore
                    metrics_class_list.append(m_instance)
                else:
                    metrics_class_list.append([m["type"]])
        return metrics_class_list

    def make_node_embeddings_args(self) -> dict:
        node_embedding_args = {
            "embedding_outfile": self.embedding_outfile(),
            "embedding_history_outfile": self.embedding_history_outfile(),
            "main_graph_args": self.main_graph_args(),
            "node_embedding_params": self.yaml["embeddings"][
                "node_embedding_params"
            ],
            "bert_columns": self.yaml["embeddings"]["bert_params"][
                "node_columns"
            ]
            if "bert_params" in self.yaml["embeddings"]
            else None,
        }
        return node_embedding_args

    #
    # tSNE stuff
    #

    def do_tsne(self) -> bool:
        return "tsne" in self.yaml["embeddings"]

    def make_tsne_args(self, graph: Graph) -> dict:
        make_tsne_args = {
            "graph": graph,
            "tsne_outfile": self.tsne_outfile(),
            "embedding_file": self.embedding_outfile(),
        }
        return make_tsne_args

    def tsne_outfile(self) -> str:
        return os.path.join(
            self.outdir(), self.yaml["embeddings"]["tsne"]["tsne_file_name"]
        )

    #
    # classifier stuff
    #

    def do_classifier(self) -> bool:
        return "classifiers" in self.yaml

    def classifier_type(self) -> str:
        return self.yaml["classifiers"]["type"]

    @catch_keyerror
    def classifiers(self) -> list:
        """From the YAML, extract a list of classifiers to be trained

        :return: list of classifiers to be trained
        """
        return self.yaml["classifiers"]

    def get_all_classifier_ids(self):
        return [c["classifier_id"] for c in self.yaml["classifiers"]]

    def get_edge_embedding_method(self, classifier: dict) -> str:
        return classifier["edge_method"]

    def classifier_history_file_name(self, classifier: dict) -> Optional[str]:
        return (
            classifier["model"]["classifier_history_file_name"]
            if "model" in classifier
            and "classifier_history_file_name" in classifier["model"]
            else None
        )

    #
    # upload stuff
    #

    def do_upload(self) -> bool:
        return "upload" in self.yaml

    def make_upload_args(self) -> dict:
        make_upload_args = {
            "local_directory": self.outdir(),
            "s3_bucket": self.yaml["upload"]["s3_bucket"],
            "s3_bucket_dir": self.yaml["upload"]["s3_bucket_dir"],
            "extra_args": self.yaml["upload"]["extra_args"]
            if "extra_args" in self.yaml["upload"]
            else None,
        }
        return make_upload_args

    #
    # deal with edge/node paths that are URLs
    #
    def deal_with_url_node_edge_paths(self):
        gd = self.yaml["graph_data"]["graph"]
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

        for item in ["node_path", "edge_path"]:
            if item in gd and is_url(gd[item]):
                url_as_filename = "".join(
                    c if c in valid_chars else "_" for c in gd[item]
                )
                outfile = os.path.join(self.outdir(), url_as_filename)
                download_file(gd[item], outfile)
                gd[item] = outfile

    #
    # applying trained model to fresh data for predictions
    #
    def do_apply_classifier(self):
        return "apply_trained_classifier" in self.yaml

    def get_classifier_id_for_prediction(self):
        return self.yaml["apply_trained_classifier"]["classifier_model_id"]

    def get_classifier_from_id(self, classifier_id: str):

        return [
            x
            for x in self.classifiers()
            if x["classifier_id"] == classifier_id
        ][0]

    def make_classifier_args(self):

        classifier_args = self.yaml["apply_trained_classifier"]

        model = self.get_classifier_from_id(
            classifier_args["classifier_model_id"]
        )
        model_filename = model["model"]["outfile"]
        classifier_args_dict = {}
        classifier_args_dict["graph"] = Graph.from_csv(
            **self.main_graph_args()
        )
        classifier_args_dict["model"] = pickle.load(
            open(
                os.path.join(self.outdir(), model_filename),
                "rb",
            ),
        )

        classifier_args_dict["node_types"] = classifier_args["link_node_types"]

        classifier_args_dict["cutoff"] = classifier_args["cutoff"]
        classifier_args_dict["output_file"] = os.path.join(
            self.outdir(), classifier_args["outfile"]
        )

        classifier_args_dict["embeddings"] = pd.read_csv(
            self.embedding_outfile(), sep=",", header=None
        )
        classifier_args_dict["edge_method"] = model["edge_method"]

        return classifier_args_dict
