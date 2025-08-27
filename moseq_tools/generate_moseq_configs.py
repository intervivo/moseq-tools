#!/usr/bin/env python
from importlib.resources import files
import click
import shutil
import ruamel.yaml as yaml



@click.group()
def cli():
    pass

def load_schema(filename):
    resource = files("moseq_tools.data") / filename
    with resource.open("r") as f:
        data = yaml.YAML(typ='safe', pure=True)
        return data.load(f)

@cli.command("extract")
@click.option("--flip-classifier-path", type=click.Path(exists=True), required=True, help="Path to the flip classifier YAML file.")
@click.option("--output-path", type=click.Path(), default="extraction-config.yaml", help="Path to save the generated extraction configuration.")
def generate_extraction_config(flip_classifier_path, output_path):
    """Generates configuration file for `moseq2-extract`."""
    print(f"Flip classifier path: {flip_classifier_path}")
    config = load_schema("extraction-config.yaml")
    config["flip_classifier"] = flip_classifier_path
    with open(output_path, "w") as f:
        data = yaml.YAML(typ='safe', pure=True)
        data.dump(config, f)

    print(f"Extraction configuration saved to {output_path}")


@cli.command("pca")
@click.option("--output-path", type=click.Path(), default="pca-config.yaml", help="Path to save the generated PCA configuration.")
def generate_model_config(output_path):
    """Generates configuration file for `moseq2-pca`."""
    resource = files("moseq_tools.data") / "pca-config.yaml"
    config_path = resource.locate()
    shutil.copyfile(config_path, output_path)

    print(f"PCA configuration saved to {output_path}")


if __name__ == "__main__":
    cli()