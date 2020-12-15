import copy
import re

from embiggen import Node2VecSequence, SkipGram, CBOW  # type: ignore
from ensmallen_graph import EnsmallenGraph  # type: ignore
from tensorflow.python.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Nadam  # type: ignore


def make_embeddings(main_graph_args: dict,
                    pos_valid_graph_args: dict,
                    embiggen_seq_args: dict,
                    node2vec_params: dict,
                    epochs: int,
                    early_stopping_args: dict,
                    model: str,
                    embedding_outfile: str,
                    model_outfile: str,
                    use_pos_valid_for_early_stopping: bool = False,
                    learning_rate: float = 0.1,
                    ) -> None:
    """Make embeddings and output embeddings and model file

    Args:
        main_graph_args: arguments passed to ensmallen_graph for graph loading
        pos_valid_graph_args: arguments passed to ensmallen_graph for positive validation graph
        embiggen_seq_args: arguments passed to Node2VecSequence() for random walk
        node2vec_params: arguments for embedding size and negative samples, passed to
            SkipGram() or CBOW()
        epochs: number of epochs to train
        use_pos_valid_for_early_stopping: should we use the positive validation graph
            for early stopping? [False]
        early_stopping_args: if we want to do early stopping, args to use
            (patience, delta, etc)
        learning_rate: learning rate passed to Nadam [0.1]
        model: SkipGram or CBOW (TODO: Glove)
        embedding_outfile: outfile for embeddings
        model_outfile: outfile for model
    Returns:
        None.

    """
    # load main graph
    graph: EnsmallenGraph = EnsmallenGraph.from_unsorted_csv(**main_graph_args)
    graph_sequence = Node2VecSequence(graph, **embiggen_seq_args)

    fit_args = {
        'steps_per_epoch': graph_sequence.steps_per_epoch,
        'callbacks': [],
        'epochs': epochs
    }

    if use_pos_valid_for_early_stopping:
        gih_params = copy.deepcopy(**main_graph_args)
        gih_params.update(**pos_valid_graph_args)
        pos_validation_graph = EnsmallenGraph.from_unsorted_csv(**gih_params)
        graph_incl_training = graph + pos_validation_graph
        gih_sequence = Node2VecSequence(graph_incl_training, **embiggen_seq_args)

        # also need to add these to be passed to model.fit()
        fit_args['validation_data'] = gih_sequence
        fit_args['validation_steps'] = gih_sequence.steps_per_epoch

    if early_stopping_args:
        es = EarlyStopping(**early_stopping_args)
        fit_args['callbacks'] = [es]

    lr = Nadam(learning_rate=learning_rate)
    word2vec_model = None
    if re.search('skipgram', model, re.IGNORECASE):
        word2vec_model = SkipGram(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                         **node2vec_params)
    elif re.search('CBOW', model, re.IGNORECASE):
        word2vec_model = CBOW(vocabulary_size=graph.get_nodes_number(), optimizer=lr,
                     **node2vec_params)
    else:
        raise NotImplemented
    ## TODO: deal with GloVe
    history = word2vec_model.fit(graph_sequence, **fit_args)
    word2vec_model.save_embedding(embedding_outfile, graph.get_node_names())
    word2vec_model.save_weights(model_outfile)
    return None
