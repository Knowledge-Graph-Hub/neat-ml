import re
from unittest import TestCase
from click.testing import CliRunner
from neat import run


class TestRun(TestCase):
    """Tests the neat.py script."""
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_run_no_yaml_file(self):
        result = self.runner.invoke(catch_exceptions=False,
                                    cli=run,
                                    args=['-y', 'doesntexist'])
        self.assertTrue(re.search('doesntexist', result.output))
        self.assertNotEqual(result.exit_code, 0)
