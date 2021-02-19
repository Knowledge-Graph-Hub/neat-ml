from unittest import TestCase
from neat.yaml_helper.yaml_helper import YamlHelper


class TestYamlHelper(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.test_yaml = "tests/resources/test.yaml"
        self.test_yaml_upload_good = 'tests/resources/test_good_upload_info.yaml'
        self.test_yaml_upload_bad = 'tests/resources/test_bad_upload_info.yaml'
        self.test_yaml_bert_tsne = 'tests/resources/test_graph_embedding_bert_tsne.yaml'
        self.test_yaml_holdouts = 'tests/resources/test_holdouts.yaml'
        self.yh = YamlHelper(self.test_yaml)

    def test_no_indir(self) -> None:
        yh = YamlHelper("tests/resources/test_no_indir.yaml")
        self.assertEqual("", yh.indir())

    def test_bad_indir(self) -> None:
        with self.assertRaises(FileNotFoundError) as context:
            YamlHelper("tests/resources/test_bad_indir.yaml").indir()

    def test_outdir(self) -> None:
        self.assertEqual("output_data", self.yh.outdir())

    def test_add_indir_to_graph_data(self):
        # emits error message to log, but continues:
        self.yh.add_indir_to_graph_data(graph_data={}, keys_to_add_indir=['not_a_key'])

    def test_do_tsne(self):
        self.assertTrue(hasattr(YamlHelper, 'do_tsne'))
        self.assertTrue(not self.yh.do_tsne())
        ybt = YamlHelper(self.test_yaml_bert_tsne)
        self.assertTrue(ybt.do_tsne())

    def test_do_embeddings(self):
        self.assertTrue(hasattr(YamlHelper, 'do_embeddings'))
        self.assertTrue(self.yh.do_embeddings())

    def test_do_classifier(self):
        self.assertTrue(hasattr(YamlHelper, 'do_classifier'))
        self.assertTrue(self.yh.do_classifier())

    def test_do_upload(self):
        self.assertTrue(hasattr(YamlHelper, 'do_upload'))
        yg = YamlHelper(self.test_yaml_upload_good)
        self.assertTrue(yg.do_upload())

    def test_make_upload_args(self):
        self.assertTrue(hasattr(YamlHelper, 'make_upload_args'))
        yg = YamlHelper(self.test_yaml_upload_good)
        self.assertDictEqual(
            yg.make_upload_args(),
            {'local_directory': 'tests/resources/test_output_data_dir/',
             's3_bucket': 'some_bucket', 's3_bucket_dir': 'some/remote/directory/'})

    def test_do_holdouts(self):
        self.assertTrue(hasattr(YamlHelper, 'do_holdouts'))
        yh = YamlHelper(self.test_yaml_holdouts)
        self.assertTrue(yh.do_holdouts())

    def test_dont_do_holdouts(self):
        self.assertTrue(hasattr(YamlHelper, 'do_holdouts'))
        yh = YamlHelper(self.test_yaml)
        self.assertTrue(not yh.do_holdouts())

    def test_make_holdouts_args(self):
        self.assertTrue(hasattr(YamlHelper, 'make_holdouts_args'))
        yh = YamlHelper(self.test_yaml_holdouts)
        mh_args = yh.make_holdouts_args()
        expected_args = {'main_graph_args': yh.main_graph_args(),
                         'output_dir': yh.outdir(),
                         'train_size': 0.8,
                         'validation': False,
                         'seed': 42,
                         'edge_types': ['biolink:interacts_with',
                                        'biolink:has_gene_product']}
        self.assertListEqual(list(mh_args.keys()), list(expected_args.keys()))
        self.maxDiff = None
        self.assertDictEqual(mh_args, expected_args)



