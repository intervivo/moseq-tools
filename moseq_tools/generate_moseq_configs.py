#!/usr/bin/env python
import click
import shutil
import ruamel.yaml as yaml
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command("extract")
@click.option("--flip-classifier-path", type=click.Path(exists=True), required=True, help="Path to the flip classifier YAML file.")
@click.option("--output-path", type=click.Path(), default="extraction-config.yaml", help="Path to save the generated extraction configuration.")
def generate_extraction_config(flip_classifier_path, output_path):
    """Generates configuration file for `moseq2-extract`."""
    print(f"Flip classifier path: {flip_classifier_path}")
    config_path = Path(__file__).resolve().parent / "extraction-config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["flip_classifier"] = flip_classifier_path
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"Extraction configuration saved to {output_path}")


@cli.command("pca")
@click.option("--output-path", type=click.Path(), default="pca-config.yaml", help="Path to save the generated PCA configuration.")
def generate_model_config(output_path):
    """Generates configuration file for `moseq2-pca`."""
    config_path = Path(__file__).resolve().parent / "pca-config.yaml"

    shutil.copyfile(config_path, output_path)

    print(f"PCA configuration saved to {output_path}")


if __name__ == "__main__":
    cli()