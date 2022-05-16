from ensmallen import Graph  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from transformers import BertModel, BertTokenizer  # type: ignore
from embiggen.pipelines import compute_node_embedding  # type: ignore


def get_node_data(file: str, sep="\t") -> pd.DataFrame:
    """Read node TSV file and return pandas dataframe

    :param file: node TSV file
    :param sep: separator
    :return:
    """
    return pd.read_csv(file, sep=sep)


def make_node_embeddings(
                         embedding_outfile: str,
                         embedding_history_outfile: str,
                         main_graph_args: dict,
                         node_embedding_params: dict,
                         bert_columns: dict,
                         bert_pretrained_model: str = "allenai/scibert_scivocab_uncased"
                         ) -> None:
    """Make embeddings and output embeddings and training history

    Args:
        embedding_outfile: outfile to write out embeddings
        embedding_history_outfile: outfile to write out training history
        main_graph_args: arguments passed to Ensmallen for graph loading
        node_embedding_params: args passed to compute_node_embeddings() in Embiggen
        bert_columns: columns containing text info to use to make embeddings from Bert
                pretrained embeddings
    Returns:
        None.

    """
    # load main graph
    if "graph_path" in main_graph_args: # Remove this extra key if present
        main_graph_args.pop("graph_path")
    graph: Graph = Graph.from_csv(**main_graph_args)
    node_embedding, training_history = compute_node_embedding(
        graph,
        **node_embedding_params
    )

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

    if not bert_embeddings.empty:
        node_embedding = pd.concat([node_embedding, bert_embeddings],
                                   axis=1,
                                   ignore_index=False)

    if not training_history.empty:
        with open(embedding_history_outfile, 'w') as f:
            f.write(training_history.to_json())

    node_embedding.to_csv(embedding_outfile, header=False)
    return None


def merge_and_write_complete_node_data(
        original_nodes_file: str, node_data: pd.DataFrame, outfile: str):
    complete_node_data = get_node_data(original_nodes_file)
    node_data = node_data.merge(complete_node_data, how="left", on="id",
                                suffixes=("", "_y")).drop("category_y", axis=1)
    node_data.to_csv(outfile, sep="\t", index=False)
