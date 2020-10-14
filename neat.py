import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("yaml_file", "-y", required=True, default="neat.yaml",
              type=click.Path(exists=True))
def run(*args, **kwargs) -> None:
    """Run a NEAT pipeline using the given YAML file [neat.yaml]

    Args:
        yaml_file: Specify the YAML file containing a list of datasets to download.

    Returns:
        None.

    """
    print("hello world")
    return None
