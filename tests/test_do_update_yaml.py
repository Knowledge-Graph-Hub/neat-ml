import os
from unittest import TestCase
from parameterized import parameterized
import tempfile
from shutil import copyfile
from neat.update_yaml.update_yaml import do_update_yaml
import yaml


def parse_yaml_file(file) -> str:
    with open(file) as f:
        return yaml.safe_load(f)


class TestDoUpdateYaml(TestCase):

    def setUp(self) -> None:
        self.pristine_yaml_file = 'tests/resources/kg-idg-neat.yaml'
        self.parsed_pristine_yaml = parse_yaml_file(self.pristine_yaml_file)
        self.yaml_file = os.path.join(tempfile.gettempdir(),
                                      os.path.basename(self.pristine_yaml_file))
        copyfile(self.pristine_yaml_file, self.yaml_file)

    def test_key_replacement(self):
        new_name = "this_new_name"
        do_update_yaml(self.yaml_file, ["name"], [new_name])
        new_yaml = parse_yaml_file(self.yaml_file)
        self.assertEqual(new_name, new_yaml['name'])

    def test_key_replacement_nested(self):
        new_graph_node_path = "this_new_path"
        do_update_yaml(self.yaml_file, ["node_path"], [new_graph_node_path])
        new_yaml = parse_yaml_file(self.yaml_file)
        self.assertEqual(new_graph_node_path,
                         new_yaml['graph_data']['graph']['node_path'])
