import re

import numpy as np
from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence, SkipGram, CBOW
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
import copy

def get_output_dir(config):
    output_dir = config['output_directory'] if 'output_directory' in config else 'output_data'


def load_graph(config_graph: dict) -> EnsmallenGraph:
    """Using a dict (parsed from a block of yaml describing how to load a graph), load
    graph using ensmallen_graph and return an EnsmallenGraph object
    """
    return EnsmallenGraph.from_unsorted_csv(**config)

def make_embeddings(config: dict) -> None:
    """Given a config dict, make embeddings. Outputs embedding file and model file
    """
    # load main graph
    graph = EnsmallenGraph.from_unsorted_csv(**config['graph'])
    graph_sequence = Node2VecSequence(graph, **config['embiggen_params']['seq_params'])

    fit_args = {
        'steps_per_epoch':graph_sequence.steps_per_epoch,
        'callbacks':[],
        'epochs': config['embiggen_params']['epochs']
    }

    if 'graph_incl_holdouts' in config:
        # make copy of config['graph']) params, overwrite with any keys in config['validation_graph']
        gih_params = copy.deepcopy(config['graph'])
        gih_params.update(config['graph_incl_holdouts'])
        graph_incl_holdouts = EnsmallenGraph.from_unsorted_csv(**gih_params)
        gih_sequence = Node2VecSequence(graph_incl_holdouts, **config['embiggen_params']['seq_params'])

        # also need to add these to be passed to model.fit()
        fit_args['validation_data']=gih_sequence
        fit_args['validation_steps']=gih_sequence.steps_per_epoch


    if 'early_stopping' in config['embiggen_params']['seq_params']:
        es = EarlyStopping(**config['embiggen_params']['seq_params']['early_stopping'])
        fit_args['callbacks'] = [es]

    lr = Nadam(config['embiggen_params']['optimizer']['learning_rate'])
    if re.search('skipgram', config['embiggen_params']['model'], re.IGNORECASE):
        model = SkipGram(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                         **config['embiggen_params']['node2vec_params'])
    elif re.search('CBOW', config['embiggen_params']['model'], re.IGNORECASE):
        model = CBOW(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                     **config['embiggen_params']['node2vec_params'])
    ## TODO: deal with GloVe
    history = model.fit(graph_sequence, **fit_args)
    np.save(os.path.join(get_output_dir(), config['embiggen_params']['embedding_file_name']), model.embedding)
    model.save_weights(os.path.join(get_output_dir(), config['embiggen_params']['weights_file']))
    return None


def make_classifier(classifier_config):
    if classifier_config['type'] in 'neural network':
        model = make_neural_net_model(classifier_config['model'])
        model_compile_parameters = classifier_config['model_compile']
        metrics = model_compile_parameters['metrics'] if 'metrics' in model_compile_parameters else None
        metrics_class_list = []
        for m in metrics:
            if m['type'].startswith('tensorflow.keras'):
                m_class = dynamically_import_class(m['type'])
                m_parameters = m['parameters']
                m_instance = m_class(**m_parameters)
                metrics_class_list.append(m_instance)
            else:
                metrics_class_list[m['type']]
        model.compile(
            loss = model_compile_parameters['loss'],
            optimizer = model_compile_parameters['optimizer'],
            metrics = metrics_class_list
        )
    elif classifier_config['type'] in 'Decision Tree':
        model = make_model(classifier_config['model'])
    elif classifier_config['type'] in 'Random Forest':
        model = make_model(classifier_config['model'])
    elif classifier_config['type'] in 'Logistic Regression':
        model = make_model(classifier_config['model'])
    return model

def model_fit(model, train_data, validation_data, parameters):
    """Takes a model, generated from make_model(), and calls .fit()

    model: output of make_model()
    data: thing that generates training and validation data
    parameters: from parsed YAML file
    """
    if 'model_fit' in parameters:
        pass
    callback_list = []
    if 'callbacks' in parameters:
        for callback in parameters['callbacks']:
            c_class = dynamically_import_class(callback['type'])
            c_instance(**callback['parameters'])
            callback_list.append(c_instance)
    del parameters['callbacks']
    model.fit(*train_data, validation_data=validation_data, **parameters, callbacks=callback_list)


def make_model(model_config: dict) -> object:
    """Make a decision tree, random forest or logistic regression model
    For neural network, use make_neural_net_model() instead
    """
    model_type = model_config['type']
    model_class = dynamically_import_class(model_type)
    model_instance = model_class()

def make_neural_net_model(model_config: dict) -> object:
    """Take the model configuration for a neural net classifier 
    from YAML and return an (uncompiled) tensorflow model
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

def make_data(embedding_file):
    embedding_model, method, train, valid = task_generator(pos_training, pos_validation, neg_training, neg_validation)
    return embedding_model, method, train, valid

def task_generator(pos_training:EnsmallenGraph, pos_validation:EnsmallenGraph, neg_training:EnsmallenGraph, neg_validation:EnsmallenGraph, train_percentage:float=0.80, seed:int=42):
    """Create new generator of tasks.

    Parameters
    ----------
    pos_training:EnsmallenGraph,
        The positive edges of the training graph.
    pos_validation:EnsmallenGraph,
        The positive edges of the validation graph.
    neg_training:EnsmallenGraph,
        The negative edges of the training graph.
    neg_validation:EnsmallenGraph,
        The negative edges of the validation graph.
    train_percentage:float=0.8,
    seed:int=42

    """
    for path in tqdm(glob(os.path.join(embedding_data_dir, "*embedding.npy")), desc="Embedding"):
        model_name = os.path.basename(path).split("_")[0]
        embedding = np.load(path, allow_pickle=True)
        for method in tqdm(EdgeTransformer.methods, desc="Methods", leave=False):

            # create graph transformer object to convert graphs into edge embeddings
            transformer = GraphTransformer(method)
            transformer.fit(embedding) # pass node embeddings to be used to create edge embeddings
            train_edges = np.vstack([ # computing edge embeddings for training graph
                transformer.transform(graph)
                for graph in (pos_training, neg_training)
            ])
            valid_edges = np.vstack([ # computing edge embeddings for validation graph
                transformer.transform(graph)
                for graph in (pos_validation, neg_validation)
            ])
            train_labels = np.concatenate([ # make labels for training graph
                np.ones(pos_training.get_edges_number()),
                np.zeros(neg_training.get_edges_number())
            ])
            valid_labels = np.concatenate([ # make labels for validation graph
                np.ones(pos_validation.get_edges_number()),
                np.zeros(neg_validation.get_edges_number())
            ])
            train_indices = np.arange(0, train_labels.size)
            valid_indices = np.arange(0, valid_labels.size)
            np.random.shuffle(train_indices) # shuffle to prevent bias caused by ordering of edge labels
            np.random.shuffle(valid_indices) # ``   ``
            train_edges = train_edges[train_indices]
            train_labels = train_labels[train_indices]
            valid_edges = valid_edges[valid_indices]
            valid_labels = valid_labels[valid_indices]
            yield model_name, method, (train_edges, train_labels), (valid_edges, valid_labels)

def dynamically_import_class(reference):
    klass = my_import(reference)
    return klass

def dynamically_import_function(reference):
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
    """Take output of make_model (a tensorflow model) and compile it
    """
    tensorflow_model.compile()
