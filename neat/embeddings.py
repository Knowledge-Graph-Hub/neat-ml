import re

import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence, SkipGram, CBOW
from matplotlib.pyplot import jet
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import copy
import os


def get_output_dir(config: dict):
    """Get output directory from config.

    Args:
        config: The config object

    Returns:
        The output directory

    """
    output_dir = config['output_directory'] if 'output_directory' in config else 'output_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def make_embeddings(config: dict) -> None:
    """Given a config dict, make embeddings. Outputs embedding file and model file

    Args:
        config: The config object

    Returns:
        None.

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
    model.save_weights(os.path.join(get_output_dir(config), config['embeddings']['model_file_name']))
    return None


def make_tsne(config: dict) -> None:
    # fail early here while debugging:
    if 'node_property_for_color' in config['embeddings']['tsne']:
        nodes = pd.read_csv(config['graph_data']['graph']['node_path'], sep='\t')
        categories = nodes[config['embeddings']['tsne']['node_property_for_color']]
        category_names = list(set(categories))
        colors = [category_names.index(i) for i in categories]
        cmap = plt.cm.get_cmap('jet', len(category_names))
        formatter = plt.FuncFormatter(lambda val, loc: category_names[val])
    else:
        colors = None
        cmap = None
        categories_idx = None

    node_embeddings = np.load(os.path.join(get_output_dir(config), config['embeddings']['embedding_file_name']))
    tsne_embeddings = TSNE(n_jobs=config['embeddings']['tsne']['n']).fit_transform(node_embeddings.data)
    x = tsne_embeddings[:, 0]
    y = tsne_embeddings[:, 1]

    plt.scatter(x, y, c=colors, cmap=cmap, **config['embeddings']['tsne']['scatter_params'])
    plt.colorbar(ticks=colors, format=formatter)
    plt.savefig(os.path.join(get_output_dir(config), config['embeddings']['tsne']['tsne_file_name']))

