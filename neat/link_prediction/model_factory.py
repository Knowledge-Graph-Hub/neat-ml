from .model import Model


class AbstractModelFactory:

    def __init__(self):
        pass

    def build(self)->Model:
        pass
