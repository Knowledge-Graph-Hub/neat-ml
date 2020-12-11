import numpy as np
import copy
import pandas as pd
from typing import Tuple
from embiggen import LinkPredictionTransformer
from ensmallen_graph import EnsmallenGraph


class Model:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def compile(self):
        pass

    def load(self, path: str) -> object:
        pass

    def save(self) -> None:
        self.model.save(self.config['model']['outfile'])

    def predict(self, X) -> np.ndarray:
        pass

    @classmethod
    def make_link_prediction_data(self,
                                  embedding_file: str,
                                  training_graph_args: dict,
                                  pos_validation_args: dict,
                                  neg_training_args: dict,
                                  neg_validation_args: dict,
                                  edge_method: str
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
        embedding = pd.read_csv(embedding_file,
                                index_col=0,
                                header=None)

        # load graphs
        graphs = {'pos_training': EnsmallenGraph.from_unsorted_csv(**training_graph_args)}
        for name, graph_args in [('pos_validation', pos_validation_args),
                                 ('neg_training', neg_training_args),
                                 ('neg_validation', neg_validation_args)]:
            these_params = copy.deepcopy(training_graph_args)
            these_params.update(graph_args)
            graphs[name] = EnsmallenGraph.from_unsorted_csv(**these_params)

        # create transformer object to convert graphs into edge embeddings
        lpt = LinkPredictionTransformer(method=edge_method)
        lpt.fit(embedding)  # pass node embeddings to be used to create edge embeddings
        train_edges, train_labels = lpt.transform(positive_graph=graphs['pos_training'],
                                                  negative_graph=graphs['neg_training'])
        valid_edges, valid_labels = lpt.transform(positive_graph=graphs['pos_validation'],
                                                  negative_graph=graphs['neg_validation'])
        return (train_edges, train_labels), (valid_edges, valid_labels)
