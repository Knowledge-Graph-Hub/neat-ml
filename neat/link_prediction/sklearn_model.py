import os
import pickle
from .model import Model


class SklearnModel(Model):
    def __init__(self, config: dict, outdir: str = None):
        super().__init__(outdir=outdir)
        self.config = config
        model_type = config["model"]["type"]
        model_class = self.dynamically_import_class(model_type)
        self.model = model_class()  # type: ignore

    def fit(self, train_data, _):
        self.model.fit(*train_data)

    def predict(self, predict_data):
        self.model.predict(*predict_data)

    def predict_proba(self, predict_data):
        self.model.predict_proba(predict_data)

    def save(self):
        with open(
            os.path.join(self.outdir, self.config["model"]["outfile"]), "wb"
        ) as f:
            pickle.dump(self.model, f)

        fn, ext = os.path.splitext(self.config["model"]["outfile"])
        model_outfile = fn + "_custom." + ext
        with open(os.path.join(self.outdir, model_outfile), "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> tuple():

        fn, ext = os.path.splitext(path)
        custom_model_filename = fn + "_custom." + ext
        with open(path, "rb") as mf1:
            generic_model_object = pickle.load(mf1)

        with open(custom_model_filename, "rb") as mf2:
            custom_model_object = pickle.load(mf2)

        # new_model = SklearnModel(self, self.outdir, self.config)
        # with open(
        #     os.path.join(
        #         self.outdir, self.config["model"]["outfile"] + "model"
        #     ),
        #     "rb",
        # ) as f:
        #     new_model = pickle.loads(self, f)
        # with open(
        #     os.path.join(self.outdir, self.config["model"]["outfile"]), "rb"
        # ) as f:
        #     new_model.model = pickle.loads(self.model, f)

        return generic_model_object, custom_model_object
