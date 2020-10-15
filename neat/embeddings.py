import re

from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence, SkipGram, CBOW
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping


def make_embeddings(config: dict) -> None:
    graph = EnsmallenGraph.from_unsorted_csv(**config['graph'])
    graph_sequence = Node2VecSequence(graph, **config['embiggen_params']['seq_params'])

    callbacks = []
    if 'early_stopping' in config['embiggen_params']['seq_params']:
        es = EarlyStopping(**config['embiggen_params']['seq_params']['early_stopping'])
        callbacks = [es]

    lr = Nadam(config['embiggen_params']['optimizer']['learning_rate'])
    if re.search('skipgram', config['embiggen_params']['model'], re.IGNORECASE):
        model = SkipGram(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                         **config['embiggen_params']['node2vec_params'])
    elif re.search('CBOW', config['embiggen_params']['model'], re.IGNORECASE):
        model = CBOW(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                     **config['embiggen_params']['node2vec_params'])
    model.summary()

    history = model.fit(
        graph_sequence,
        steps_per_epoch=graph_sequence.steps_per_epoch,
        callbacks=callbacks,
        epochs=config['embiggen_params']['epochs']
    )

    return None


def make_classifier(config: dict) -> None:
    pass
