import copy
import re

from embiggen import Node2VecSequence, SkipGram, CBOW  # type: ignore
from ensmallen_graph import EnsmallenGraph  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tensorflow.python.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Nadam  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from transformers import BertModel, BertTokenizer  # type: ignore


def get_node_data(file: str, sep="\t") -> pd.DataFrame:
    """Read node TSV file and return pandas dataframe

    :param file: node TSV file
    :param sep: separator
    :return:
    """
    return pd.read_csv(file, sep=sep)


def make_graph_embeddings(main_graph_args: dict,
                          pos_valid_graph_args: dict,
                          embiggen_seq_args: dict,
                          node2vec_params: dict,
                          epochs: int,
                          early_stopping_args: dict,
                          model: str,
                          embedding_outfile: str,
                          model_outfile: str,
                          embedding_history_outfile: str,
                          use_pos_valid_for_early_stopping: bool = False,
                          learning_rate: float = 0.1,
                          bert_columns: list = None,
                          bert_pretrained_model: str = "allenai/scibert_scivocab_uncased"
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
        embedding_history_outfile: outfile for history
        bert_columns: list of columns from bert_node_file to embed
    Returns:
        None.

    """
    # load main graph
    graph: EnsmallenGraph = EnsmallenGraph.from_unsorted_csv(**main_graph_args)
    graph_sequence = Node2VecSequence(graph, **embiggen_seq_args)

    # embed columns with BERT first (if we're gonna)
    bert_embeddings = pd.DataFrame()
    if bert_columns:
        bert_model = BertModel.from_pretrained(bert_pretrained_model,
                                               output_hidden_states=True)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        bert_model.eval()
        all_bert_embeddings = bert_model.embeddings.word_embeddings.weight.data.numpy()

        node_data = get_node_data(main_graph_args['node_path'])

        node_text = [
            " ".join([str(row[col]) for col in bert_columns])
            for index, row in tqdm(node_data.iterrows(), "extracting text from nodes")
        ]
        node_text_tokenized = [bert_tokenizer.encode(
            this_text,  # Sentence to encode
            # add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_tensors='np'
        ) for this_text in tqdm(node_text, "tokenzing text")]
        node_text_tensors = [
            np.mean(all_bert_embeddings[ids.flatten()], axis=0)
            for ids in tqdm(node_text_tokenized, "extracting embeddings for tokens")]

        bert_embeddings = pd.DataFrame(node_text_tensors, index=graph.get_node_names())

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
        raise NotImplementedError(f"{model} isn't implemented yet")

    ## TODO: deal with GloVe
    history = word2vec_model.fit(graph_sequence, **fit_args)

    if embedding_history_outfile:
        with open(embedding_history_outfile, 'w') as f:
            f.write(history.to_json())

    these_embeddings = pd.DataFrame(word2vec_model.embedding,
                                    index=graph.get_node_names())

    if not bert_embeddings.empty:
        these_embeddings = pd.concat([these_embeddings, bert_embeddings],
                                     axis=1,
                                     ignore_index=False)

    these_embeddings.to_csv(embedding_outfile, header=False)
    word2vec_model.save_weights(model_outfile)
    return None


def merge_and_write_complete_node_data(
        original_nodes_file: str, node_data: pd.DataFrame, outfile: str):
    complete_node_data = get_node_data(original_nodes_file)
    node_data = node_data.merge(complete_node_data, how="left", on="id",
                                suffixes=("", "_y")).drop("category_y", axis=1)
    node_data.to_csv(outfile, sep="\t", index=False)
