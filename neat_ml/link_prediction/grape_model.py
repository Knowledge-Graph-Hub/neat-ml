"""Grape models for link prediction."""

from .model import Model
from grape import edge_prediction, edge_prediction_evaluation


class GrapeModel(Model):
    """Grape (Embiggen) link prediction model class."""

    def __init__(self, config: dict, outdir: str = None):
        """Initialize GrapeModel class."""
        super().__init__(outdir=outdir)
        self.config = config
        model_type = config["classifier_type"]
        model_class = self.dynamically_import_class(model_type)
        self.model = model_class()  # type: ignore

    def fit(self, train_data, test_data):
        """Fit model."""
        return self.model.fit(train_data, test_data)

    def predict_proba(self, predict_data):
        """Predict probability."""
        return self.model.predict_proba(predict_data)

    # Grape methods don't currently support loading 
    # classifiers.

    def load(self, path: str) -> tuple():  # type: ignore
         """Return error regarding load function."""
         print(f"Looking at {path}...")
         raise NotImplementedError("Grape methods do not" 
                                    " currently support loading.")
