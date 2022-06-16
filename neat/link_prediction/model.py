import os
import pickle

import numpy as np  # type: ignore
import copy
import pandas as pd  # type: ignore
from typing import Optional, Tuple, Union
from grape.embedding_transformers import EdgePredictionTransformer, GraphTransformer  # type: ignore
from grape import Graph  # type: ignore
import importlib


class Model:
    def __init__(self, outdir=None):
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

    def fit(self, X, y):
        pass

    def compile(self):
        pass

    def load(self, path: str) -> object:
        pass

    def save(self):
        with open(
            os.path.join(self.outdir, self.config["outfile"]), "wb"
        ) as f:
            pickle.dump(self, f)

    def predict(self, predict_data) -> np.ndarray:
        return self.model.predict(predict_data) # type: ignore

    def predict_proba(self, X) -> np.ndarray:
        pass

    @classmethod
    def make_train_valid_data(
        self,
        embedding_file: str,
        training_graph_args: dict,
        validation_args: Optional[dict] = None,
        training_args: Optional[dict] = None,
        edge_method: str = "Average",
    ) -> Tuple[Tuple, Tuple]:
        """Prepare training and validation data for training link prediction classifers

        Args:
            embedding_file: path to embedding file for nodes in graph
            training_graph_args: Ensmallen arguments to load training graph
            validation_args: Ensmallen arguments to load positive and negative validation graphs
            training_args: Ensmallen arguments to load negative training graph
            edge_method: edge embedding method to use (average, L1, L2, etc)
        Returns:
            A tuple of tuples

        """

        embedding = pd.read_csv(embedding_file, index_col=0, header=None)

        # load graphs
        graphs = {"pos_training": Graph.from_csv(**training_graph_args)}
        is_directed = graphs["pos_training"].is_directed()
        graph_filepaths = {
            "pos_validation": "pos_edge_filepath",
            "neg_training": "neg_edge_filepath",
            "neg_validation": "neg_edge_filepath"
        }
        for name, graph_args in [
            ("pos_validation", validation_args),
            ("neg_training", training_args),
            ("neg_validation", validation_args),
        ]:
            if not graph_args:
                if name in ["neg_training", "neg_validation"]:
                    neg_edge_number = graphs["pos_training"].get_edges_number()
                    if not is_directed:
                        neg_edge_number = neg_edge_number * 2

                    graphs[name] = (graphs["pos_training"]).sample_negative_graph(
                        number_of_negative_samples=neg_edge_number
                    )
                else:
                    these_params = copy.deepcopy(training_graph_args)
                    if "directed" not in these_params.keys():
                        these_params["directed"] = training_graph_args[
                            "directed"
                        ]
                    graphs[name] = Graph.from_csv(**these_params)
            else:
                these_params = copy.deepcopy(training_graph_args)
                these_params["edge_path"] = graph_args[graph_filepaths[name]]
                graphs[name] = Graph.from_csv(**these_params)
                                                    
        # create transformer object to convert graphs into edge embeddings
        lpt = EdgePredictionTransformer(method=edge_method)

        lpt.fit(
            embedding
        )  # pass node embeddings to be used to create edge embeddings

        train_edges, train_labels = lpt.transform(
            positive_graph=graphs["pos_training"],
            negative_graph=graphs["neg_training"],
        )
        valid_edges, valid_labels = lpt.transform(
            positive_graph=graphs["pos_validation"],
            negative_graph=graphs["neg_validation"],
        )
        return (train_edges, train_labels), (valid_edges, valid_labels)

    @classmethod
    def make_edge_embedding_for_predict(
        self,
        embedding_file: str,
        edge_method: str,
        source_destination_list
    ) -> np.ndarray:
        """Prepare training and validation data for training link prediction classifers

        Args:
            embedding_file: path to embedding file for nodes in graph
            trained_graph_args: EnsmallenGraph arguments to load training graph
            edge_method: edge embedding method to use (average, L1, L2, etc)
        Returns:
            A NumPy Array embeddings that represent prediction edges.

        """

        embedding = pd.read_csv(embedding_file, index_col=0, header=None)

        # Create graphtransformer object for edge embeddings
        gt = GraphTransformer(method=edge_method)

        gt.fit(embedding)

        edge_embedding_for_predict = gt.transform(
            graph=source_destination_list
        )

        return edge_embedding_for_predict

    @classmethod
    def dynamically_import_class(self, reference) -> object:
        """Dynamically import a class based on its reference.

        Args:
            reference: The reference or path for the class to be imported.

        Returns:
            The imported class

        """
        klass = self.my_import(reference)
        return klass

    @classmethod
    def dynamically_import_function(self, reference) -> object:
        """Dynamically import a function based on its reference.

        Args:
            reference: The reference or path for the function to be imported.

        Returns:
            The imported function

        """
        module_name = ".".join(reference.split(".")[0:-1])
        function_name = reference.split(".")[-1]
        f = getattr(importlib.import_module(module_name), function_name)
        return f

    @classmethod
    def my_import(self, name):
        components = name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod
