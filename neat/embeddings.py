import re

from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence, SkipGram, CBOW
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
import copy

def get_output_dir(config):
    output_dir = config['output_directory'] if 'output_directory' in config else 'output_data'


def make_embeddings(config: dict) -> None:
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
    model.summary()

    history = model.fit(graph_sequence, **fit_args)
    return None

def make_model(classifier_config: dict) -> object:
    """Take the classifier arg parsed from YAML and return an (uncompiled) tensorflow model
    """
    pass

def compile_model(tensorflow_model: obj, config: dict) -> None:
    # tensorflow_model.compile()
    pass

def make_classifier(config: dict) -> None:
    """Train classifier
    """
    make_model(config)
