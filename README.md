# Network Embedding All the Things (NEAT)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=alert_status)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT) [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=coverage)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT)

NEAT is a flexible pipeline for:
- Parsing a graph serialization
- Generating embeddings
- Training Classifiers
- Making predictions
- Creating well formatted output and metrics for the predictions

## Requirements

This pipeline has [Embiggen](https://github.com/monarch-initiative/embiggen) and [tensorflow](https://github.com/tensorflow/tensorflow) as major dependencies.
Please install the versions of `tensorflow`, `CUDA`, and `cudnn` compatible with your system and with each other prior to installing NEAT.

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
git clone https://github.com/Knowledge-Graph-Hub/NEAT.git
cd NEAT
pip install .
```

## Running NEAT
```
neat run --config tests/resources/test.yaml # example
neat run --config [your yaml]
```

The pipeline is driven by a YAML file (e.g. `tests/resources/test.yaml`), which contains all parameters needed to complete the pipeline.
This includes hyperparameters for machine learning and also things like files/paths to output results.
Please note that the input paths in `node_path` and `edge_path`
```
graph_data:
  graph:
    node_path: 
    edge_path: 
```
may be local filepaths **OR** URLs.

A diagram explaining the design a bit is [here](https://app.diagrams.net/#G1XLKYf9ZiBfWmjfAIeI9yYv_CycE8GmIQ).

If you are uploading to AWS/S3, see here for configuring AWS credentials:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
