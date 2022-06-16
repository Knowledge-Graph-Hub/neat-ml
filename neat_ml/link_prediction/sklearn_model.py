import pickle
from .model import Model


class SklearnModel(Model):
    def __init__(self, config: dict, outdir: str = None):
        super().__init__(outdir=outdir)
        self.config = config
        model_type = config["classifier_type"]
        model_class = self.dynamically_import_class(model_type)
        self.model = model_class()  # type: ignore

    def fit(self, train_data, test_data):
        return self.model.fit(train_data, test_data)

    def predict_proba(self, predict_data):
        return self.model.predict_proba(predict_data)

    def load(self, path: str) -> tuple(): # type: ignore
        with open(path, "rb") as mf1:
            model_object = pickle.load(mf1)
        return model_object  # , custom_model_object
