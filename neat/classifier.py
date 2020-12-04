import copy
import importlib
import os
from typing import Tuple

import numpy as np
import tensorflow
from embiggen import GraphTransformer, EdgeTransformer
from ensmallen_graph import EnsmallenGraph

from neat.embeddings import get_output_dir

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# def make_classifier(classifier_config: dict) -> object:
#     """Make a model for a classifier based on a given
#     set of configurations.
#
#     Args:
#         classifier_config: The config corresponding to 'classifiers'
#
#     Returns:
#         model: The compiled model
#
#     """
#     if 'neural network' in classifier_config['type']:
#         model = make_neural_net_model(classifier_config, classifier_config['model'])
#         model_compile_parameters = classifier_config['model_compile']
#         metrics = model_compile_parameters['metrics'] if 'metrics' in model_compile_parameters else None
#         metrics_class_list = []
#         for m in metrics:
#             if m['type'].startswith('tensorflow.keras'):
#                 m_class = dynamically_import_class(m['type'])
#                 m_parameters = m['parameters']
#                 m_instance = m_class(**m_parameters)
#                 metrics_class_list.append(m_instance)
#             else:
#                 metrics_class_list.append([m['type']])
#         model.compile(
#             loss = model_compile_parameters['loss'],
#             optimizer = model_compile_parameters['optimizer'],
#             metrics = metrics_class_list
#         )
#     elif classifier_config['type'] in 'Decision Tree':
#         model = make_model(classifier_config, classifier_config['model'])
#     elif classifier_config['type'] in 'Random Forest':
#         model = make_model(classifier_config, classifier_config['model'])
#     elif classifier_config['type'] in 'Logistic Regression':
#         model = make_model(classifier_config, classifier_config['model'])
#     return model


def model_fit(config, model, train_data, validation_data, classifier) -> object:
    """Takes a model, generated from make_model(), and calls .fit()

    Args:
        config: the config of the parent
        model: output of make_model()
        data: thing that generates training and validation data
        classifier: classifier config from parsed YAML file

    Returns:
        The model object

    """
    try:
        classifier_params = classifier['model_fit']['parameters']
    except KeyError:
        classifier_params = {}

    callback_list = []
    if 'callbacks' in classifier_params:
        for callback in classifier_params['callbacks']:
            c_class = dynamically_import_class(callback['type'])
            c_params = callback['parameters'] if 'parameters' in callback else {}
            c_instance = c_class(**c_params)
            callback_list.append(c_instance)
        del classifier_params['callbacks']
    if isinstance(model, tensorflow.python.keras.engine.sequential.Sequential):
        model.fit(*train_data, validation_data=validation_data, **classifier_params, callbacks=callback_list)
        model.save(os.path.join(get_output_dir(config), classifier['model']['outfile']))
    else:
        model.fit(*train_data, **classifier_params)
        with open(os.path.join(get_output_dir(config), classifier['model']['outfile']), 'wb') as f:
            pickle.dump(model, f)


def make_model(config, model_config: dict) -> object:
    """Make a decision tree, random forest or logistic regression model
    For neural network, use make_neural_net_model() instead

    Args:
        config: the classifier config
        model_config: the model config

    Returns:
        The model

    """
    model_type = model_config['type']
    model_class = dynamically_import_class(model_type)
    model_instance = model_class()
    return model_instance


def make_neural_net_model(config, model_config: dict) -> object:
    """Take the model configuration for a neural net classifier
    from YAML and return an (uncompiled) tensorflow model

    Args:
        config: the classifier config
        model_config: the model config

    Returns:
        The model

    """
    model_type = model_config['type']
    model_class = dynamically_import_class(model_type)

    model_layers = []
    for layer in model_config['layers']:
        layer_type = layer['type']
        layer_class = dynamically_import_class(layer_type)
        parameters = layer['parameters']
        layer_instance = layer_class(**parameters)
        model_layers.append(layer_instance)
    model_instance = model_class()
    for l in model_layers:
        model_instance.add(l)
    return model_instance


# def make_data(config) -> Tuple[Tuple, Tuple]:
#     """Prepare data for training and validation.
#
#     Args:
#         config: the config
#
#     Returns:
#         A tuple of tuples
#
#     """
#     embedding = np.load(
#         os.path.join(get_output_dir(config), config['embeddings']['embedding_file_name']))
#
#     # load graphs
#     graphs = {'pos_training': EnsmallenGraph.from_unsorted_csv(**config['graph_data']['graph'])}
#     for this_graph in ['pos_validation', 'neg_training', 'neg_validation']:
#         these_params = copy.deepcopy(config['graph_data']['graph'])
#         these_params.update(config['graph_data'][this_graph])
#         graphs[this_graph] = EnsmallenGraph.from_unsorted_csv(**these_params)
#
#     # create graph transformer object to convert graphs into edge embeddings
#     transformer = GraphTransformer(config['classifier']['edge_method'])
#     transformer.fit(embedding)  # pass node embeddings to be used to create edge embeddings
#     train_edges = np.vstack([  # computing edge embeddings for training graph
#         transformer.transform(graph)
#         for graph in (graphs['pos_training'], graphs['neg_training'])
#     ])
#     valid_edges = np.vstack([ # computing edge embeddings for validation graph
#         transformer.transform(graph)
#         for graph in (graphs['pos_validation'], graphs['neg_validation'])
#     ])
#     train_labels = np.concatenate([ # make labels for training graph
#         np.ones(graphs['pos_training'].get_edges_number()),
#         np.zeros(graphs['neg_training'].get_edges_number())
#     ])
#     valid_labels = np.concatenate([ # make labels for validation graph
#         np.ones(graphs['pos_validation'].get_edges_number()),
#         np.zeros(graphs['neg_validation'].get_edges_number())
#     ])
#     train_indices = np.arange(0, train_labels.size)
#     valid_indices = np.arange(0, valid_labels.size)
#     np.random.shuffle(train_indices) # shuffle to prevent bias caused by ordering of edge labels
#     np.random.shuffle(valid_indices) # ``   ``
#     train_edges = train_edges[train_indices]
#     train_labels = train_labels[train_indices]
#     valid_edges = valid_edges[valid_indices]
#     valid_labels = valid_labels[valid_indices]
#     return (train_edges, train_labels), (valid_edges, valid_labels)


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


def compile_model(tensorflow_model: object, config: dict) -> None:
    """Take output of make_model (a tensorflow model) and compile
    the model with given arguments.

    Args:
        tensorflow_model: The tensorflow model instance
        config: the config

    Returns:
        None.

    """
    tensorflow_model.compile()
