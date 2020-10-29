#!/bin/bash

wget http://purl.obolibrary.org/obo/go.json -o go.json
kgx transform --input-format obojson --output-format tsv --output go go.json