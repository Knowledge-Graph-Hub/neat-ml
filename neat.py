import click
import yaml


@click.group()
def cli():
    pass


@cli.command()
@click.option("yaml_file", "-y", required=True, default="neat.yaml",
              type=click.Path(exists=True))
def run(yaml_file) -> None:
    """Run a NEAT pipeline using the given YAML file [neat.yaml]\f

    Args:
        yaml_file: Specify the YAML file containing a list of datasets to download.

    Returns:
        None.

    """
    yaml_args = parse_yaml(yaml_file)
    return None


def parse_yaml(file: str) -> object:
    with open(file, 'r') as stream:
        return yaml.safe_load(stream)


if __name__ == "__main__":
    cli()
