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
graph_args = go_yaml['graph_data']['graph']
graph_args['edge_path'] = go_yaml['graph_data']['pos_validation']['edge_path']

validation_graph = EnsmallenGraph.from_unsorted_csv(**graph_args)

# these BP graph edges are the ones we want to evaluate
edges_to_eval = list(zip(validation_graph.get_source_names(),
                 validation_graph.get_destination_names()))

edge_transform = GraphTransformer(go_yaml['classifier']['edge_method'])
edge_transform.fit(
    np.load(os.path.join("output_data", go_yaml['embeddings']['embedding_file_name'])))
edges_to_eval_emb = edge_transform.transform(validation_graph)

mlp_file = [c for c in go_yaml['classifier']['classifiers'] if c['type'] == 'neural network'][0]['model']['outfile']
mlp = tf.keras.models.load_model(os.path.join("output_data", mlp_file))

edges_to_eval_predict = mlp.predict(edges_to_eval_emb, batch_size=1048)
edges_to_eval_predict_sorted = pd.DataFrame({
    "pred": edges_to_eval_predict.flatten(),
    "subject": [t[0] for t in edges_to_eval],
    "object": [t[1] for t in edges_to_eval]
}).sort_values(by=["pred"], ascending=True)

edges_to_eval_predict_sorted.to_csv("sorted_sco_edges.tsv", sep='\t', index=False)
