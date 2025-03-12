#!/usr/bin/env python
import os
import click
import logging
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist


def load_df(df_path, usage_threshold=0):
    df = pd.read_csv(df_path, index_col=0)
    df = df[df['usage'] > usage_threshold]
    return df


@click.command()
@click.argument("dset1_df_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("dset2_df_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", "-o", type=click.Path(), default=os.getcwd(), help="Output path")
@click.option("--usage-threshold", type=float, default=0.005, help="Minimum usage threshold")
def main(dset1_df_path, dset2_df_path, output_path, usage_threshold):
    """Read in two moseq2_df_stats.csv dataframes and compute pairwise cosine distance
    between different drug groups (assuming the groups have been added to the
    moseq2-index.yaml file).

    Args:
        dset1_df_path: Path to first dataset's moseq2_df_stats.csv
        dset2_df_path: Path to second dataset's moseq2_df_stats.csv (found in the
            comparisons folder in the directory of the first dataset)
        output_path: Path to save output
        usage_threshold: Minimum usage threshold
    """
    log_path = Path(output_path) / "compute_cosine_distances.log"
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode="w")

    dset1_df = load_df(dset1_df_path, usage_threshold)
    dset2_df = load_df(dset2_df_path, usage_threshold)

    # compute usage by group
    logging.info("Averaging syllable usage by group")
    dset1_usage_by_group = (
        dset1_df.groupby(["group", "syllable"])["usage"].mean().unstack(fill_value=0)
    )
    dset2_usage_by_group = (
        dset2_df.groupby(["group", "syllable"])["usage"].mean().unstack(fill_value=0)
    )

    # reindex columns so that they match
    all_unique_columns = set(dset1_usage_by_group.columns) | set(dset2_usage_by_group.columns)
    dset1_usage_by_group = dset1_usage_by_group.reindex(all_unique_columns, axis=1, fill_value=0)
    dset2_usage_by_group = dset2_usage_by_group.reindex(all_unique_columns, axis=1, fill_value=0)

    logging.info("Computing pairwise cosine distance")
    pairwise_distances = cdist(dset1_usage_by_group.values, dset2_usage_by_group.values, metric="cosine")

    # turn into df
    pairwise_distances_df = pd.DataFrame(
        pairwise_distances, index=dset1_usage_by_group.index, columns=dset2_usage_by_group.index
    )

    logging.info("Saving pairwise distances to CSV")
    save_path = Path(output_path) / "pairwise_distances.csv" if output_path else "pairwise_distances.csv"
    pairwise_distances_df.to_csv(save_path)
    logging.info(f"Saved pairwise distances to {save_path}")
    print(f"Saved pairwise distances to {save_path}")
    print(pairwise_distances_df)


if __name__ == "__main__":
    main()
