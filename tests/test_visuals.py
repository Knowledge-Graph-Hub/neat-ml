"""Test visuals."""
import os
from unittest import TestCase

from click.testing import CliRunner

from neat_ml.cli import run


class TestVisuals(TestCase):
    """Test visuals Class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up."""
        pass

    def setUp(self) -> None:
        """Set up."""
        self.runner = CliRunner()
        self.expected_tsne_file = (
            "tests/resources/test_output_data_dir/test_tsne.png"
        )
        self.expected_fullplot_file = (
            "tests/resources/test_output_data_dir/test_plots.png"
        )
        for filepath in [self.expected_tsne_file, self.expected_fullplot_file]:
            if os.path.exists(filepath):
                print(f"removing existing test tsne file {filepath}")
                os.unlink(filepath)

    def test_make_tsne(self):
        """Test for making a tSNE using grape."""
        self.runner.invoke(
            catch_exceptions=False,
            cli=run,
            args=["--config", "tests/resources/test_for_tsne.yaml"],
        )

        self.assertTrue(os.path.exists(self.expected_tsne_file))

    # Disabled for now, as this should really be called in the
    # same way as the tsne generator
    # def test_make_all_plots(self):

    #     result = self.runner.invoke(
    #         catch_exceptions=False,
    #         cli=run,
    #         args=["--config", "tests/resources/test_for_plots.yaml"],
    #     )

    #     tsne_kwargs = {"graph": g,
    #         "tsne_outfile": self.expected_fullplot_file,
    #         "embedding_file": 'tests/resources/test_embeddings.csv',
    #         }
    #     make_all_plots(**tsne_kwargs)

    #     self.assertTrue(os.path.exists(self.expected_fullplot_file))
