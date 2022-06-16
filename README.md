# Network Embedding All the Things (NEAT)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=alert_status)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT) [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=coverage)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT)

NEAT is a flexible pipeline for:

- Parsing a graph serialization
- Generating embeddings
- Training Classifiers
- Making predictions
- Creating well formatted output and metrics for the predictions

## Quick Start

```
pip install neat-ml
neat run --config neat_quickstart.yaml
```

NEAT will write graph embeddings to a new `quickstart_output` directory.

## Requirements

This pipeline has [grape](https://github.com/AnacletoLAB/grape) as a major dependency.

Methods from [tensorflow](https://github.com/tensorflow/tensorflow) are supported, but are not installed as dependencies to avoid version conflicts.

Please install the versions of `tensorflow`, `CUDA`, and `cudnn` compatible with your system and with each other prior to installing NEAT if you wish to use these methods.

On Linux, this may be easiest using `conda` as follows:

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O anaconda.sh
bash ./anaconda.sh -b
echo "export PATH=\$PATH:$HOME/anaconda3/bin" >> $HOME/.bashrc
conda init
conda install cudnn
conda install tensorflow
```

## Installation

```
pip install neat-ml
```

## Running NEAT

```
neat run --config tests/resources/test.yaml # example
neat run --config [your yaml]
```

The pipeline is driven by a YAML file (e.g. `tests/resources/test.yaml`), which contains all parameters needed to complete the pipeline.
The contents and expected values for this file are defined by the [neat-ml-schema](https://github.com/Knowledge-Graph-Hub/neat-ml-schema).

This includes hyperparameters for machine learning and also things like files/paths to output results.
Specify paths to node and edge files:

```
GraphDataConfiguration:
  graph:
    directed: False
    node_path: path/to/nodes.tsv
    edge_path: path/to/edges.tsv
```

If the graph data is in a compressed file and/or a remote location (e.g., on KG-Hub), one or more URLs may be specified in the `source_data` parameter:

```
GraphDataConfiguration:
  source_data:
    files:
      - path: https://kg-hub.berkeleybop.io/kg-obo/bfo/2019-08-26/bfo_kgx_tsv.tar.gz
        desc: "This is BFO, your favorite basic formal ontology, now in graph form."
      - path: https://someremoteurl.com/graph2.tar.gz
        desc: "This is some other graph - it may be useful."

```

A diagram explaining the design a bit is [here](https://app.diagrams.net/#G1XLKYf9ZiBfWmjfAIeI9yYv_CycE8GmIQ).

If you are uploading to AWS/S3, [see here for configuring AWS credentials:](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)

## Credits

Developed by Deepak Unni, Justin Reese, J. Harry Caufield, and Harshad Hegde.
