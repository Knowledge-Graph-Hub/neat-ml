import os
import yaml


def parse_yaml(file: str) -> object:
    with open(file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


class YamlHelper:
    """
    Class to parse yaml and extract args for methods
    """

    def __init__(self, config: str):
        self.default_output_dir = 'output_data'
        self.yaml = parse_yaml(config)

    def output_dir(self):
        """Get output directory from config.

        Args:
            config: The config object

        Returns:
            The output directory

        """
        output_dir = self.yaml['output_directory'] if 'output_directory' in self.yaml \
            else self.default_output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    #
    # graph stuff
    #
    def main_graph_args(self) -> dict:
        return self.yaml['graph_data']['graph']

    def pos_val_graph_args(self) -> dict:
        return self.yaml['graph_data']['pos_validation']

    def neg_val_graph_args(self) -> dict:
        return self.yaml['graph_data']['neg_validation']

    def neg_train_graph_args(self) -> dict:
        return self.yaml['graph_data']['neg_training']

    #
    # embedding stuff
    #

    def do_embeddings(self) -> bool:
        return 'embeddings' in self.yaml

    def embedding_outfile(self) -> str:
        return os.path.join(self.output_dir(),
                            self.yaml['embeddings']['embedding_file_name'])

    def model_outfile(self) -> str:
        return os.path.join(self.output_dir(),
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
            'learning_rate': self.yaml['embeddings']['embiggen_params']['optimizer']['learning_rate']
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
            make_tsne_args['node_file'] = self.yaml['graph_data']['graph']['node_path']
            make_tsne_args['node_property_for_color'] = \
                self.yaml['embeddings']['tsne']['node_property_for_color']
        return make_tsne_args

    def tsne_outfile(self) -> str:
        return os.path.join(self.output_dir(),
                            self.yaml['embeddings']['tsne']['tsne_file_name'])

    #
    # classifier stuff
    #

    def classifiers(self) -> list:
        """From the YAML, extract a list of classifiers to be trained

        :return: list of classifiers to be trained
        """
        return self.yaml['classifier']['classifiers']

    def edge_embedding_method(self) -> str:
        return self.yaml['classifier']['edge_method']

