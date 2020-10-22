import re

import numpy as np
from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence, SkipGram, CBOW
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
import copy
import os


def get_output_dir(config):
    output_dir = config['output_directory'] if 'output_directory' in config else 'output_data' 
    if not os.path(output_dir):
        os.makedirs(output_dir, exist_ok=true)
    return output_dir


def make_embeddings(config: dict) -> None:
    """Given a config dict, make embeddings. Outputs embedding file and model file
    """
    # load main graph
    graph = EnsmallenGraph.from_unsorted_csv(**config['graph_data']['graph'])
    graph_sequence = Node2VecSequence(graph, **config['embeddings']['embiggen_params']['seq_params'])

    fit_args = {
        'steps_per_epoch': graph_sequence.steps_per_epoch,
        'callbacks': [],
        'epochs': config['embeddings']['embiggen_params']['epochs']
    }

    if 'use_pos_valid_for_early_stopping' in config and config['embeddings']['use_pos_valid_for_early_stopping']:
        gih_params = copy.deepcopy(config['graph_data']['graph'])
        gih_params.update(config['graph_data']['pos_validation_graph'])
        pos_validation_graph = EnsmallenGraph.from_unsorted_csv(**gih_params)
        graph_incl_training = graph + pos_validation_graph
        gih_sequence = Node2VecSequence(graph_incl_training, **config['embeddings']['embiggen_params']['seq_params'])

        # also need to add these to be passed to model.fit()
        fit_args['validation_data']=gih_sequence
        fit_args['validation_steps']=gih_sequence.steps_per_epoch

    if 'early_stopping' in config['embeddings']['embiggen_params']['seq_params']:
        es = EarlyStopping(**config['embeddings']['embiggen_params']['seq_params']['early_stopping'])
        fit_args['callbacks'] = [es]

    lr = Nadam(config['embeddings']['embiggen_params']['optimizer']['learning_rate'])
    if re.search('skipgram', config['embeddings']['embiggen_params']['model'], re.IGNORECASE):
        model = SkipGram(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                         **config['embeddings']['embiggen_params']['node2vec_params'])
    elif re.search('CBOW', config['embeddings']['embiggen_params']['model'], re.IGNORECASE):
        model = CBOW(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                     **config['embeddings']['embiggen_params']['node2vec_params'])
    ## TODO: deal with GloVe
    history = model.fit(graph_sequence, **fit_args)
    np.save(os.path.join(get_output_dir(config), config['embeddings']['embedding_file_name']), model.embedding)
    model.save_weights(os.path.join(get_output_dir(config), config['embeddings']['weights_file']))
    return None

