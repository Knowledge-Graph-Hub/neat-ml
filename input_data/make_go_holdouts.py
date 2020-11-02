from ensmallen_graph import EnsmallenGraph
train_percentage = 0.8
seed = 42
graph = EnsmallenGraph.from_unsorted_csv(
    name="go-json",
    edge_path="go_edges.tsv",
    sources_column="subject",
    destinations_column="object",
    edge_types_column='edge_label',
    directed=False,
    node_path="go_nodes.tsv",
    nodes_column='id',
    node_types_column='category',
    default_node_type='biolink:NamedThing'
)

reduced_graph = graph.remove(singletons=True)
pos_training, pos_validation = reduced_graph.connected_holdout(
    train_size=train_percentage,
    edge_types=['biolink:subclass_of', 'biolink:part_of'],
    random_state=seed)

pos_training.dump_edges("go_edges_training.tsv")
pos_validation.dump_edges("go_edges_validation.tsv")

