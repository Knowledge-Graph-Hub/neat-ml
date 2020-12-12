import os
import pickle
from .model import Model


class SklearnModel(Model):

    def __init__(self, config: dict, outdir: str = None):
        super().__init__(outdir=outdir)
        self.config = config
        model_type = config['model']['type']
        model_class = self.dynamically_import_class(model_type)
        self.model = model_class()

    def fit(self, train_data, _):
        self.model.fit(*train_data)

    def save(self):
        with open(os.path.join(self.outdir, self.config['model']['outfile'], 'wb')) as f:
             pickle.dump(self.model, f)

