"""Grape models for link prediction."""

from .model import Model

from grape import Graph
import pandas as pd
import numpy as np
from typing import Optional, Union, List


class GrapeModel(Model):
    """Grape (Embiggen) link prediction model class."""

    def __init__(self, config: dict, outdir: str = None):
        """Initialize GrapeModel class."""
        super().__init__(outdir=outdir)
        self.config = config
        model_type = config["classifier_type"]
        model_class = self.dynamically_import_class(model_type)
        self.model = model_class()  # type: ignore
        self.is_fit = False

    def fit(self, 
            graph: Graph, 
            support: Optional[Graph] = None,
            node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
            node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
            edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None):
        """Fit model.

        Based on the Ensmallen abstract 
        edge prediction model class.
        """
        fit_model = self.model.fit(graph=graph,
                                    support=support,
                                    node_features=node_features,
                                    node_type_features=node_type_features,
                                    edge_features=None
        )
        self.is_fit = True
        return fit_model

    def predict_proba(self, predict_data):
        """Predict probability.

        The source method includes optional
        parameters not implemented here.
        """
        return self.model.predict_proba(predict_data)

    # Grape methods don't currently support loading
    # classifiers.

    def load(self, path: str) -> tuple():  # type: ignore
        """Return error regarding load function.

        This may be supported in the future,
        but for now we don't load these models.
        """
        print(f"Looking at {path}...")
        raise NotImplementedError("Grape methods do not"
                                  " currently support loading.")
