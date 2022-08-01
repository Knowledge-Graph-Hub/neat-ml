"""method_names.py - constants for names passed to CLI"""

import grape

NN_NAMES = ["mlp",
            "multilayer perceptron",
            "nn",
            "neural network"
            ]

LR_NAMES = ["lr",
            "logistic regression"
            ]

gmodels = (
    grape.get_available_models_for_edge_prediction()["model_name"]
).tolist()
gmodels = [mname.lower() for mname in gmodels]
GRAPE_LP_CLASS_NAMES = [gmodels] + \
                       [model + " classifier" for model in gmodels]