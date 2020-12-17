import copy
import re

import torch
from embiggen import Node2VecSequence, SkipGram, CBOW  # type: ignore
from ensmallen_graph import EnsmallenGraph  # type: ignore
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Nadam  # type: ignore
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer


def get_node_data(file: str, sep="\t") -> pd.DataFrame:
    """Read node TSV file and return pandas dataframe

    :param file:
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
        bert_columns: list of columns from bert_node_file to embed
    Returns:
        None.

    """
    # embed columns with BERT first (if we're gonna)
    bert_embeddings = {}
    if bert_columns:
        bert_model = BertModel.from_pretrained(bert_pretrained_model,
                                               output_hidden_states=True)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        bert_model.eval()

        node_data = get_node_data(main_graph_args['node_path'])

        for _, row in tqdm(node_data.iterrows(),
                           "making BERT embeddings of columns: " + " ".join(bert_columns),
                           total=node_data.shape[0]):
            node_text = "".join([row[col] for col in bert_columns])
            bert_embeddings[row['id']] = get_embedding(bert_model, bert_tokenizer, node_text)

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
        raise NotImplementedError(f"{model} isn't implemented yet")

    ## TODO: deal with GloVe
    word2vec_model.fit(graph_sequence, **fit_args)

    word2vec_model.save_embedding(embedding_outfile, graph.get_node_names())
    word2vec_model.save_weights(model_outfile)
    return None


def get_embedding(bert_model, tokenizer, text) -> np.array:
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.

    :param bert_model
    :param tokenizer
    :text  text for which we want an embedding
    '''

    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors.
    encoded_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids = encoded_dict['input_ids']

    bert_model.eval()

    # Run the text through the model and get the hidden states.
    bert_outputs = bert_model(input_ids)

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = bert_model(input_ids)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [13 x 1 x <sentence length> x 768]

    # Select the embeddings from the second to last layer.
    # `token_vecs` is a tensor with shape [<sent length> x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # Convert to numpy array.
    sentence_embedding = sentence_embedding.detach().numpy()

    # If `word` was provided, compute an embedding for those tokens.
    return sentence_embedding
