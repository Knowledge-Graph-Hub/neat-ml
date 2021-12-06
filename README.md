# Network Embedding All the Things (NEAT)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=alert_status)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT) [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Knowledge-Graph-Hub_NEAT&metric=coverage)](https://sonarcloud.io/dashboard?id=Knowledge-Graph-Hub_NEAT)

NEAT is a flexible pipeline for:
- Parsing a graph serialization
- Generating embeddings
- Training Classifiers
- Making predictions
- Creating well formatted output and metrics for the predictions

To install:

```
git clone https://github.com/Knowledge-Graph-Hub/NEAT.git
cd NEAT
# install tensorflow - if you need GPUs, consult this reference to install the version appropriate for you:
# https://www.tensorflow.org/install/source#gpu
pip install .
```

To run:
```
neat run --config tests/resources/test.yaml # example
neat run --config [your yaml]
```

This pipeline has [Embiggen](https://github.com/monarch-initiative/embiggen) and [tensorflow](https://github.com/tensorflow/tensorflow) as major dependencies.

The pipeline is driven by a YAML file (e.g. `tests/resources/test.yaml`), which contains all parameters needed to complete the pipeline.
This includes hyperparameters for machine learning and also things like files/paths to output results.

A diagram explaining the design a bit is [here](https://app.diagrams.net/#G1XLKYf9ZiBfWmjfAIeI9yYv_CycE8GmIQ).

If you are uploading to AWS/S3, see here for configuring AWS credentials:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
