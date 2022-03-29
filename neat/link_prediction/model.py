import os
import pickle

import numpy as np  # type: ignore
import copy
import pandas as pd  # type: ignore
from typing import Optional, Tuple, Union
from embiggen import LinkPredictionTransformer  # type: ignore
from ensmallen import Graph  # type: ignore
import sklearn  # type: ignore
import tensorflow  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

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
        pass

    def predict(self, X) -> np.ndarray:
        pass

    def predict_proba(self, X) -> np.ndarray:
        pass

    @classmethod
    def make_train_valid_data(
        self,
        embedding_file: str,
        training_graph_args: dict,
        edge_method: str,
        pos_validation_args: Optional[dict] = None,
        neg_training_args: Optional[dict] = None,
        neg_validation_args: Optional[dict] = None,
    ) -> Tuple[Tuple, Tuple]:
        """Prepare training and validation data for training link prediction classifers

        Args:
            embedding_file: path to embedding file for nodes in graph
            training_graph_args: EnsmallenGraph arguments to load training graph
            pos_validation_args: EnsmallenGraph arguments to load positive validation graph
            neg_training_args: EnsmallenGraph arguments to load negative training graph
            neg_validation_args: EnsmallenGraph arguments to load negative validation graph
            edge_method: edge embedding method to use (average, L1, L2, etc)
        Returns:
            A tuple of tuples

        """
        embedding = pd.read_csv(embedding_file, index_col=0, header=None)

        # TODO: re-evaluate this behavior
        # If a positive validation set isn't provided, we may
        # want to just skip the validation.
        # Same with evaluating against negative train/validate.

        # load graphs
        graphs = {"pos_training": Graph.from_csv(**training_graph_args)}
        is_directed = graphs["pos_training"].is_directed()
        for name, graph_args in [
            ("pos_validation", pos_validation_args),
            ("neg_training", neg_training_args),
            ("neg_validation", neg_validation_args),
        ]:
            if not graph_args:
                if name in ["neg_training", "neg_validation"]:
                    neg_edge_number = graphs["pos_training"].get_edges_number()
                    if not is_directed:
                        neg_edge_number = neg_edge_number * 2

                    graphs[name] = (graphs["pos_training"]).sample_negatives(
                        negatives_number=neg_edge_number
                    )
                else:
                    these_params = copy.deepcopy(training_graph_args)
                    graphs[name] = Graph.from_csv(**these_params)
            else:
                graphs[name] = Graph.from_csv(**graph_args)

        # create transformer object to convert graphs into edge embeddings
        lpt = LinkPredictionTransformer(method=edge_method)

        lpt.fit(
            embedding
        )  # pass node embeddings to be used to create edge embeddings
        # TODO: Save lpt object(?)
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
    def make_link_predictions(
        self,
        embedding_file: str,
        trained_graph_args: dict,
        edge_method: str,
        pos_validation_args: Optional[dict] = None,
        neg_training_args: Optional[dict] = None,
        neg_validation_args: Optional[dict] = None,
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
        # Get 'directed' value from trained_graph_args
        neg_training_args["directed"] = trained_graph_args["directed"]

        # load graphs
        graphs = {
            "trained_graph": Graph.from_csv(**trained_graph_args),
            "negative_graph": Graph.from_csv(**neg_training_args),
        }

        # create transformer object to convert graphs into edge embeddings
        lpt = LinkPredictionTransformer(method=edge_method)
        # TODO: Instead of fit, use embeddings that was previously exported.
        lpt.fit(
            embedding
        )  # pass node embeddings to be used to create edge embeddings
        with open("test.pickle", "wb") as file:
            pickle.dump(lpt, file)

        predict_edges, _ = lpt.transform(
            positive_graph=graphs["trained_graph"],
            negative_graph=graphs["negative_graph"],
        )
        import pdb

        pdb.set_trace()

        return predict_edges

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
