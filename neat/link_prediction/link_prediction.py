import sklearn
import tensorflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import importlib


def dynamically_import_class(reference) -> object:
    """Dynamically import a class based on its reference.

    Args:
        reference: The reference or path for the class to be imported.

    Returns:
        The imported class

    """
    klass = my_import(reference)
    return klass


def dynamically_import_function(reference) -> object:
    """Dynamically import a function based on its reference.

    Args:
        reference: The reference or path for the function to be imported.

    Returns:
        The imported function

    """
    module_name = '.'.join(reference.split('.')[0:-1])
    function_name = reference.split('.')[-1]
    f = getattr(importlib.import_module(module_name), function_name)
    return f


def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

