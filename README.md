# Neural network Embedding All the Things (NEAT)

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
pip install .
```

To run:
```
neat run --config tests/resources/test.yaml # example
neat run --config [your yaml]
```

This pipeline has [KGX](https://github.com/biolink/kgx) and [Embiggen](https://github.com/monarch-initiative/embiggen) as major dependencies.

The pipeline is driven by a YAML file (e.g. `tests/resources/test.yaml`), which contains all parameters needed to complete the pipeline.
This includes hyperparameters for machine learning and also things like files/paths to output results.

A diagram explaining the design a bit is [here](https://app.diagrams.net/#G1XLKYf9ZiBfWmjfAIeI9yYv_CycE8GmIQ).
