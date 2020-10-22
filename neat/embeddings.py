import re

import numpy as np
from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence, SkipGram, CBOW, EdgeTransformer, GraphTransformer
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
import copy
import os


def get_output_dir(config):
    output_dir = config['output_directory'] if 'output_directory' in config else 'output_data'
    return output_dir


def make_embeddings(config: dict) -> None:
    """Given a config dict, make embeddings. Outputs embedding file and model file
    """
    # load main graph
    graph = EnsmallenGraph.from_unsorted_csv(**config['graph_data']['graph'])
    graph_sequence = Node2VecSequence(graph, **config['embiggen_params']['seq_params'])

    fit_args = {
        'steps_per_epoch': graph_sequence.steps_per_epoch,
        'callbacks': [],
        'epochs': config['embiggen_params']['epochs']
    }

    if 'use_training_for_early_stopping' in config and config['use_training_for_early_stopping']:
        gih_params = copy.deepcopy(config['graph_data']['graph'])
        gih_params.update(config['pos_validation_graph'])
        pos_validation_graph = EnsmallenGraph.from_unsorted_csv(**gih_params)
        graph_incl_training = graph + pos_validation_graph
        gih_sequence = Node2VecSequence(graph_incl_training, **config['embiggen_params']['seq_params'])

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

