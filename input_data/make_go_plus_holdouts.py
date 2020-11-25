import os

from ensmallen_graph import EnsmallenGraph
from shutil import which

train_percentage = 0.8
seed = 42

go_plus_json_file = "go_plus.json"
go_plus_edges_file = "go_plus_edges.tsv"
go_plus_nodes_file = "go_plus_nodes.tsv"


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    # from whichcraft import which
    return which(name) is not None


if not is_tool("kgx"):
    raise Exception("Need to install KGX! (pip install kgx)") # might also need to roll bmt back to v 0.1.1

if not os.path.exists(go_plus_json_file):
    os.system("wget http://purl.obolibrary.org/obo/go/extensions/go-plus.json -O go_plus.json")

if not os.path.exists(go_plus_edges_file) or not os.path.exists(go_plus_nodes_file):
    os.system("kgx transform --input-format obojson --output-format tsv --output go_plus go.json")

for edges in (
    ['biolink:subclass_of'],
):
    edges_string = "_".join(edges)
    os.makedirs(edges_string, exist_ok=True)

    graph = EnsmallenGraph.from_unsorted_csv(
        edge_path=go_plus_edges_file,
        sources_column="subject",
        destinations_column="object",
        edge_types_column='edge_label',
        directed=False,
        node_path=go_plus_nodes_file,
        nodes_column='id',
        node_types_column='category',
        default_node_type='biolink:NamedThing'
    )

    reduced_graph = graph.remove(singletons=True)
    pos_training, pos_validation = reduced_graph.connected_holdout(
        train_size=train_percentage,
        edge_types=edges,
        random_state=seed)

    # make negative graph
    neg_training, neg_validation = reduced_graph.sample_negatives(
       random_state=seed,
       only_from_same_component=True,
       negatives_number=graph.get_edges_number(),
    ).random_holdout(random_state=seed, train_size=train_percentage)

    reduced_graph.dump_nodes(os.path.join(edges_string, f"go_plus_nodes_training.tsv"))

    pos_training.dump_edges(os.path.join(edges_string, f"go_plus_edges_training.tsv"))
    pos_validation.dump_edges(os.path.join(edges_string, f"go_plus_edges_validation.tsv"))
    neg_training.dump_edges(os.path.join(edges_string, f"go_plus_edges_neg_training.tsv"))
    neg_validation.dump_edges(os.path.join(edges_string, f"go_plus_edges_neg_validation.tsv"))

    os.system(f"sed -i '.bak' 's/$/\tlabel/' {edges_string}/go_plus_edges_neg_training.tsv")
    os.system(f"sed -i '.bak' 's/$/\tlabel/' {edges_string}/go_plus_edges_neg_validation.tsv")
