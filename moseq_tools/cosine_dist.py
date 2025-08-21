#!/usr/bin/env python
import os
import click
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist


def load_df(df_path, usage_threshold=0):
    df = pd.read_csv(df_path, index_col=0)
    df = df[df["usage"] > usage_threshold]
    return df


def compute_subject_level_distances(
    dset_a_df: pd.DataFrame, dset_b_df: pd.DataFrame, all_unique_columns
) -> list:
    """Compute subject-level cosine distances between groups across two datasets.

    For each group in dset_a_df, compute cosine distances to each group in dset_b_df
    at the subject level (via uuid) and return a list of rows compatible with the boxplot_df schema.

    Args:
        dset_a_df: First dataset dataframe containing columns ['group', 'uuid', 'syllable', 'usage']
        dset_b_df: Second dataset dataframe with the same columns
        all_unique_columns: Union of syllable columns used to align matrices

    Returns:
        List[dict]: rows of the form {"group1": <group in A>, "group2": <group in B>, "distance": <float>}
    """
    rows = []
    for group_a in dset_a_df["group"].unique():
        filtered_a = dset_a_df[dset_a_df["group"] == group_a]
        filtered_a = filtered_a.pivot_table(
            index="uuid", columns="syllable", values="usage", fill_value=0
        ).reindex(all_unique_columns, axis=1, fill_value=0)

        for group_b in dset_b_df["group"].unique():
            filtered_b = dset_b_df[dset_b_df["group"] == group_b]
            filtered_b = filtered_b.pivot_table(
                index="uuid", columns="syllable", values="usage", fill_value=0
            ).reindex(all_unique_columns, axis=1, fill_value=0)

            dists = cdist(filtered_a.values, filtered_b.values, metric="cosine").mean(
                axis=1
            )
            for _dist, _uuid in zip(dists, filtered_a.index):
                rows.append(
                    {
                        "group1": group_a,
                        "group2": group_b,
                        "cosine_distance": _dist,
                        "uuid": _uuid,
                    }
                )

    return rows


@click.command()
@click.argument("dset1_df_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("dset2_df_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-path", "-o", type=click.Path(), default=os.getcwd(), help="Output path"
)
@click.option(
    "--usage-threshold", type=float, default=0.005, help="Minimum usage threshold"
)
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
    path = Path(output_path) if output_path else Path.cwd()
    log_path = path / "compute_cosine_distances.log"
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
    all_unique_columns = set(dset1_usage_by_group.columns) | set(
        dset2_usage_by_group.columns
    )
    dset1_usage_by_group = dset1_usage_by_group.reindex(
        all_unique_columns, axis=1, fill_value=0
    )
    dset2_usage_by_group = dset2_usage_by_group.reindex(
        all_unique_columns, axis=1, fill_value=0
    )

    logging.info("Computing pairwise cosine distance")
    pairwise_distances = cdist(
        dset1_usage_by_group.values, dset2_usage_by_group.values, metric="cosine"
    )

    # turn into df
    pairwise_distances_df = pd.DataFrame(
        pairwise_distances,
        index=dset1_usage_by_group.index,
        columns=dset2_usage_by_group.index,
    )

    logging.info("Saving pairwise distances to CSV")
    save_path = path / "pairwise_distances.csv"
    pairwise_distances_df.to_csv(save_path)
    logging.info(f"Saved pairwise distances to {save_path}")
    print(f"Saved pairwise distances to {save_path}")
    print(pairwise_distances_df)

    # compute subject-level distances
    # one direction (dset1 vs dset2) and reverse direction (dset2 vs dset1)
    boxplot_df = [
        *compute_subject_level_distances(dset1_df, dset2_df, all_unique_columns),
        *compute_subject_level_distances(dset2_df, dset1_df, all_unique_columns),
    ]
    boxplot_df = pd.DataFrame(boxplot_df)

    # Create boxplot
    fig = plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=boxplot_df, x="group2", y="cosine_distance", hue="group1")
    ax.set(
        title="Cosine distances between groups",
        xlabel="Comparison group",
        ylabel="Cosine distance",
    )
    ax.legend(title="Drug group", bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path / "pairwise_distances_boxplot.png")
    boxplot_df.to_csv(path / "subject-level_pairwise_distances.csv")


if __name__ == "__main__":
    main()
