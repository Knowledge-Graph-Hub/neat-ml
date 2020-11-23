from ensmallen_graph import EnsmallenGraph
from embiggen import GraphTransformer
import numpy as np
import pandas as pd
import yaml
import os
import tensorflow as tf

go_yaml_file = "go.yaml"
os.chdir("..")  # paths to files are from root dir

with open(go_yaml_file, 'r') as stream:
    go_yaml = yaml.load(stream, Loader=yaml.FullLoader)

mlp_file = [c for c in go_yaml['classifier']['classifiers'] if c['type'] == 'neural network'][0]['model']['outfile']
mlp = tf.keras.models.load_model(os.path.join("output_data", mlp_file))

node_data = pd.read_csv('input_data/go_nodes.tsv', sep='\t')
node_data = node_data.filter(['id', 'name'])

#
# positive validation edges
#
pos_graph_args = go_yaml['graph_data']['graph']
pos_graph_args['directed'] = True
pos_graph_args['edge_path'] = go_yaml['graph_data']['pos_validation']['edge_path']
pos_validation_graph = EnsmallenGraph.from_unsorted_csv(**pos_graph_args)
pos_edges = list(zip(pos_validation_graph.get_source_names(),
                     pos_validation_graph.get_destination_names()))

pos_edge_transform = GraphTransformer(go_yaml['classifier']['edge_method'])
pos_edge_transform.fit(
    np.load(os.path.join("output_data", go_yaml['embeddings']['embedding_file_name'])))
pos_edges_to_eval_emb = pos_edge_transform.transform(pos_validation_graph)

pos_valid_predict = mlp.predict(pos_edges_to_eval_emb, batch_size=1048)
pos_valid_predict_sorted = pd.DataFrame({
    "pred": pos_valid_predict.flatten(),
    "subject": [t[0] for t in pos_edges],
    "object": [t[1] for t in pos_edges]
}).sort_values(by=["pred"], ascending=True)

# add GO CURIE names
pos_valid_predict_sorted = \
    pos_valid_predict_sorted.\
        merge(how='left',
              right=node_data,
              left_on='subject',
              right_on='id').drop('id', axis=1).\
        merge(how='left',
              right=node_data,
              left_on='object',
              right_on='id').drop('id', axis=1)
pos_valid_predict_sorted = \
    pos_valid_predict_sorted.rename(columns={'name_x': 'subject_name',
                                             'name_y': 'object_name'})
pos_valid_predict_sorted.to_csv("pos_sco_edges.tsv", sep='\t', index=False)

#
# negative validation edges
#
neg_graph_args = go_yaml['graph_data']['graph']
neg_graph_args['directed'] = True
neg_graph_args['edge_path'] = go_yaml['graph_data']['neg_validation']['edge_path']
neg_validation_graph = EnsmallenGraph.from_unsorted_csv(**neg_graph_args)
neg_edges = list(zip(neg_validation_graph.get_source_names(),
                     neg_validation_graph.get_destination_names()))

neg_edge_transform = GraphTransformer(go_yaml['classifier']['edge_method'])
neg_edge_transform.fit(
    np.load(os.path.join("output_data", go_yaml['embeddings']['embedding_file_name'])))
neg_edges_to_eval_emb = neg_edge_transform.transform(neg_validation_graph)

neg_valid_predict = mlp.predict(neg_edges_to_eval_emb, batch_size=1048)
neg_valid_predict_sorted = pd.DataFrame({
    "pred": neg_valid_predict.flatten(),
    "subject": [t[0] for t in neg_edges],
    "object": [t[1] for t in neg_edges]
}).sort_values(by=["pred"], ascending=False)

neg_valid_predict_sorted = \
    neg_valid_predict_sorted.\
        merge(how='left',
              right=node_data,
              left_on='subject',
              right_on='id').drop('id', axis=1).\
        merge(how='left',
              right=node_data,
              left_on='object',
              right_on='id').drop('id', axis=1)
neg_valid_predict_sorted = \
    neg_valid_predict_sorted.rename(columns={'name_x': 'subject_name',
                                             'name_y': 'object_name'})

neg_valid_predict_sorted.to_csv("neg_sco_edges.tsv", sep='\t', index=False)
