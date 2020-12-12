# Neural network Embedding All the Things (NEAT)

NEAT is a flexible pipeline for:
- Parsing a graph serialization
- Generating embeddings
- Training Classifiers
- Making predictions
- Creating well formatted output and metrics for the predictions

This pipeline has [KGX](https://github.com/biolink/kgx) and [Embiggen](https://github.com/monarch-initiative/embiggen) as major dependencies.

The pipeline is driven by a YAML file, `neat.yaml`, which contains all parameters needed to complete the pipeline.
This includes hyperparameters for machine learning and also things like files/paths to output results.

A diagram explaining the design a bit is [here](https://app.diagrams.net/#G1XLKYf9ZiBfWmjfAIeI9yYv_CycE8GmIQ).
