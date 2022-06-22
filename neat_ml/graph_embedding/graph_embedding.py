import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import transformers  # type: ignore
from grape import Graph  # type: ignore
from grape.embedders import embed_graph  # type: ignore
from tqdm.auto import tqdm  # type: ignore


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
    bert_pretrained_model: str = "allenai/scibert_scivocab_uncased",
) -> None:
    """Make embeddings and output embeddings and training history.

    :param embedding_outfile: Outfile to write out embeddings.
    :param embedding_history_outfile: Outfile to write out embedding history.
    :param main_graph_args: Arguments passed to Ensmallen for graph loading.
    :param node_embedding_params: Arguments passed to Embiggen.
    :param bert_columns: Columns containing text info to 
        use to make embeddings from Bert pretrained embeddings
    :param bert_pretrained_model: _description_, defaults to "allenai/scibert_scivocab_uncased"
    :return: None
    """
    # load main graph
    graph: Graph = Graph.from_csv(**main_graph_args)

    node_embedding = embed_graph(
        graph, node_embedding_params["method_name"], node_embedding_params
    ).get_node_embedding_from_index(0)

    # embed columns with BERT first (if we're gonna)
    bert_embeddings = pd.DataFrame()
    if bert_columns:
        bert_model = transformers.BertModel.from_pretrained(
            bert_pretrained_model, output_hidden_states=True
        )
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            bert_pretrained_model
        )
        bert_model.eval()
        all_bert_embeddings = (
            bert_model.embeddings.word_embeddings.weight.data.numpy()
        )

        node_data = get_node_data(main_graph_args["node_path"])

        node_text = [
            " ".join([str(row[col]) for col in bert_columns])
            for index, row in tqdm(
                node_data.iterrows(), "extracting text from nodes"
            )
        ]
        node_text_tokenized = [
            bert_tokenizer.encode(
                this_text,  # Sentence to encode
                # add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                return_tensors="np",
            )
            for this_text in tqdm(node_text, "tokenzing text")
        ]
        node_text_tensors = [
            np.mean(all_bert_embeddings[ids.flatten()], axis=0)
            for ids in tqdm(
                node_text_tokenized, "extracting embeddings for tokens"
            )
        ]

        bert_embeddings = pd.DataFrame(
            node_text_tensors, index=graph.get_node_names()
        )

    if not bert_embeddings.empty:
        node_embedding = pd.concat(
            [node_embedding, bert_embeddings], axis=1, ignore_index=False
        )

    node_embedding.to_csv(embedding_outfile, header=False)
    return None


def merge_and_write_complete_node_data(
    original_nodes_file: str, node_data: pd.DataFrame, outfile: str
):
    complete_node_data = get_node_data(original_nodes_file)
    node_data = node_data.merge(
        complete_node_data, how="left", on="id", suffixes=("", "_y")
    ).drop("category_y", axis=1)
    node_data.to_csv(outfile, sep="\t", index=False)
