"""Test YAML update."""
import os
import tempfile
from shutil import copyfile
from unittest import TestCase

import yaml

from neat_ml.update_yaml.update_yaml import do_update_yaml


def parse_yaml_file(file) -> str:
    """Parse YAML file."""
    with open(file) as f:
        return yaml.safe_load(f)


class TestDoUpdateYaml(TestCase):
    """Test YAML update."""

    def setUp(self) -> None:
        """Set up."""
        self.pristine_yaml_file = "tests/resources/kg-idg-neat.yaml"
        self.parsed_pristine_yaml = parse_yaml_file(self.pristine_yaml_file)
        self.yaml_file = os.path.join(
            tempfile.gettempdir(), os.path.basename(self.pristine_yaml_file)
        )
        copyfile(self.pristine_yaml_file, self.yaml_file)

    def test_key_replacement_nested(self):
        """Test YAML updates."""
        new_graph_node_path = "this_new_path"
        do_update_yaml(
            self.yaml_file,
            ["GraphDataConfiguration:graph:node_path"],
            [new_graph_node_path],
        )
        new_yaml = parse_yaml_file(self.yaml_file)
        self.assertEqual(
            new_graph_node_path,
            new_yaml["GraphDataConfiguration"]["graph"]["node_path"],
        )
