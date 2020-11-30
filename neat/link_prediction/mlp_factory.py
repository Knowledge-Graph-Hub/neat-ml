from .model_factory import AbstractModelFactory
from .mlp import MLPModel


class MLPModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def build(self) -> MLPModel:
        pass
