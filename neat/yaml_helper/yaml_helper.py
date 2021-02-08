import functools
import logging
import os
import yaml


def parse_yaml(file: str) -> dict:
    with open(file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def catch_keyerror(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyError as e:
            print("can't find key in YAML: ", e, "(possibly harmless)")
            return None
    return func


class YamlHelper:
    """
    Class to parse yaml and extract args for methods
    """

    def __init__(self, config: str):
        self.default_outdir = 'output_data'
        self.default_indir = ''
        self.yaml: dict = parse_yaml(config)

    def indir(self):
        """Get input directory from config.

        Returns:
            The input directory

        """
        if 'input_directory' in self.yaml:
            indir = self.yaml['input_directory']
            if not os.path.exists(indir):
                raise Exception(f"Can't find input dir {indir}")
        else:
            indir = self.default_indir
        return indir

    def outdir(self):
        """Get output directory from config.

        Returns:
            The output directory

        """
        outdir = self.yaml['output_directory'] if 'output_directory' in self.yaml \
            else self.default_outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return outdir

    def add_indir_to_graph_data(self, graph_data: dict,
                                keys_to_add_indir: list = ['node_path', 'edge_path']) -> dict:
        """
        :param graph_data - parsed yaml
        :param keys_to_add_indir: what keys to add indir to
        :return:
        """
        for k in keys_to_add_indir:
            if k in graph_data:
                graph_data[k] = os.path.join(self.indir(), graph_data[k])
            else:
                logging.error(f"Can't find key {k} in graph_data! Skipping")
        return graph_data

    #
    # graph stuff
    #
    def main_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(self.yaml['graph_data']['graph'])

    @catch_keyerror
    def pos_val_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(self.yaml['graph_data']['pos_validation'])

    @catch_keyerror
    def neg_val_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(self.yaml['graph_data']['neg_validation'])

    @catch_keyerror
    def neg_train_graph_args(self) -> dict:
        return self.add_indir_to_graph_data(self.yaml['graph_data']['neg_training'])

    #
    # embedding stuff
    #

    def do_embeddings(self) -> bool:
        return 'embeddings' in self.yaml

    def embedding_outfile(self) -> str:
        return os.path.join(self.outdir(),
                            self.yaml['embeddings']['embedding_file_name'])

    def model_outfile(self) -> str:
        return os.path.join(self.outdir(),
                            self.yaml['embeddings']['model_file_name'])

    def make_embedding_args(self) -> dict:
        make_embedding_args = {
            'main_graph_args': self.main_graph_args(),
            'pos_valid_graph_args': self.pos_val_graph_args(),
            'embiggen_seq_args': self.yaml['embeddings']['embiggen_params']['seq_params'],
            'node2vec_params': self.yaml['embeddings']['embiggen_params']['node2vec_params'],
            'epochs': self.yaml['embeddings']['embiggen_params']['epochs'],
            'early_stopping_args': self.yaml['embeddings']['embiggen_params']['early_stopping'],
            'model': self.yaml['embeddings']['embiggen_params']['model'],
            'embedding_outfile': self.embedding_outfile(),
            'model_outfile': self.model_outfile(),
            'use_pos_valid_for_early_stopping': 'use_pos_valid_for_early_stopping' in self.yaml,
            'learning_rate': self.yaml['embeddings']['embiggen_params']['optimizer']['learning_rate'],
            'bert_columns': self.yaml['embeddings']['bert_params']['node_columns']
            if 'bert_params' in self.yaml['embeddings'] else None
        }
        return make_embedding_args

    #
    # tSNE stuff
    #

    def do_tsne(self) -> bool:
        return 'tsne' in self.yaml['embeddings']

    def make_tsne_args(self) -> dict:
        make_tsne_args = {
            'tsne_outfile': self.tsne_outfile(),
            'embedding_file': self.embedding_outfile(),
            'num_processors': self.yaml['embeddings']['tsne']['n'],
            'scatter_params': self.yaml['embeddings']['tsne']['scatter_params'],
            'color_nodes': 'node_property_for_color' in self.yaml['embeddings']['tsne']
        }
        if 'node_property_for_color' in self.yaml['embeddings']['tsne']:
            make_tsne_args['node_file'] = os.path.join(self.indir(), self.yaml['graph_data']['graph']['node_path'])
            make_tsne_args['node_property_for_color'] = \
                self.yaml['embeddings']['tsne']['node_property_for_color']
        return make_tsne_args

    def tsne_outfile(self) -> str:
        return os.path.join(self.outdir(),
                            self.yaml['embeddings']['tsne']['tsne_file_name'])

    #
    # classifier stuff
    #

    def do_classifier(self) -> bool:
        return 'classifier' in self.yaml

    def classifier_type(self) -> str:
        return self.yaml['classifier']['type']

    def classifiers(self) -> list:
        """From the YAML, extract a list of classifiers to be trained

        :return: list of classifiers to be trained
        """
        return self.yaml['classifier']['classifiers']

    def edge_embedding_method(self) -> str:
        return self.yaml['classifier']['edge_method']

