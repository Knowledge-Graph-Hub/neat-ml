import os
from unittest import TestCase, skip
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

    def test_key_replacement_nested(self):
        new_graph_node_path = "this_new_path"
        do_update_yaml(self.yaml_file, ["GraphDataConfiguration:graph:node_path"], [new_graph_node_path])
        new_yaml = parse_yaml_file(self.yaml_file)
        self.assertEqual(new_graph_node_path,
                         new_yaml['GraphDataConfiguration']['graph']['node_path'])

    def test_key_multiple_replacements(self):
        keys = ['Target:target_path', 
                'GraphDataConfiguration:graph:node_path', 
                'GraphDataConfiguration:graph:edge_path',
                'Upload:s3_bucket_dir']
        values = ['20211202/graph_ml_artifacts',
                    '20211202/merged-kg_nodes.tsv',
                    '20211202/merged-kg_edges.tsv', 
                    'kg-idg/20211202/graph_ml_artifacts']
        do_update_yaml(self.yaml_file, keys=keys, values=values)
        new_yaml = parse_yaml_file(self.yaml_file)
        self.assertEqual(values[0], new_yaml[keys[0]])
        self.assertEqual(values[1], new_yaml[keys[1]])
        self.assertTrue('graph' in new_yaml['graph_data'])
        self.assertNotEqual(None, new_yaml['graph_data']['graph'])
        self.assertEqual(values[2], new_yaml['graph_data']['graph']['node_path'])
        self.assertEqual(values[3], new_yaml['graph_data']['graph']['edge_path'])
        self.assertEqual(values[4], new_yaml['upload']['s3_bucket_dir'])
