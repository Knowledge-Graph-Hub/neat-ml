#!/bin/bash

wget http://purl.obolibrary.org/obo/go.json -o input_data/go.json
kgx transform --input-format obojson --output-format tsv --output input_data/go input_data/go.json