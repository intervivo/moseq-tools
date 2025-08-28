#!/usr/bin/env python
import importlib.resources as pkg_resources
from pathlib import Path
import click
import shutil
import ruamel.yaml as yaml



@click.group()
def cli():
    pass


def load_schema(filename):
    # open_text returns a file-like object from inside the package
    with pkg_resources.open_text("moseq_tools.data", filename) as f:
        data = yaml.YAML(typ="safe", pure=True)
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
    with pkg_resources.path("moseq_tools.data", "pca-config.yaml") as p:
        config_path = Path(p)

    shutil.copyfile(config_path, output_path)
    print(f"PCA configuration saved to {output_path}")


if __name__ == "__main__":
    cli()