#!/usr/bin/env python
import h5py
import click
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from toolz.curried import pluck
from scipy.ndimage import maximum_filter1d
from scipy.spatial.distance import pdist, squareform
from toolz import groupby, compose, partial, valfilter, valmap
from moseq2_viz.model.util import compute_behavioral_statistics


def load_syllables(path: str):
    logging.info(f"Loading syllables from {path}")
    model = joblib.load(path)
    uuids = model["keys"]

    if "label_resamples" not in model:
        raise ValueError(
            "MoSeq model does not contain label_resamples key - model needs to be trained with resamples (e.g. --save-every 1)"
        )

    resamples = np.array(list(model["label_resamples"].values()))
    arr_labels = []
    for i in range(resamples.shape[1]):
        arr_labels.append(np.array(resamples[:, i].tolist()).astype("int16"))
    arr_labels = np.hstack(arr_labels)

    # combine labels into one long-form array
    return uuids, arr_labels[:, np.all(arr_labels >= 0, axis=0)]


def load_pcs(path: str, uuids):
    logging.info(f"Loading PCs from {path}")
    with h5py.File(path, "r") as f:
        pcs = [f["scores"][u][3:, :10] for u in uuids]
    pcs = np.vstack(pcs)
    zpcs = (pcs - np.nanmean(pcs, axis=0, keepdims=True)) / np.nanstd(
        pcs, axis=0, keepdims=True
    )
    return zpcs


def compute_pc_similarity(pcs, syllable_labels, max_label: int = 100):
    grouped_pcs = groupby(lambda k: int(k[0]), zip(syllable_labels, pcs))
    grouped_pcs = valmap(compose(list, pluck(1)), grouped_pcs)
    grouped_pcs = valmap(np.stack, grouped_pcs)

    mean_syllable_pc = valmap(partial(np.nanmean, axis=0), grouped_pcs)

    pc_arr = np.zeros((max_label, 10))
    for k, v in mean_syllable_pc.items():
        pc_arr[k] = v

    dist = squareform(pdist(pc_arr, metric="cosine"))
    return 1 - dist


def count_syllables(data):
    _count = partial(np.bincount, minlength=100)
    return np.apply_along_axis(_count, 0, data)


def get_low_entropy_frame_counts(data, pcs=None, threshold=0.25):
    eps = 1e-12
    count_transitions = partial(np.bincount, minlength=2)

    # filter out frames that are associated with transitions
    transitions = np.diff(data.astype("int16"), axis=0) != 0
    trans_freq = np.apply_along_axis(count_transitions, 0, transitions)
    trans_freq = trans_freq / (np.sum(trans_freq, axis=0, keepdims=True) + eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        trans_ent = np.nan_to_num(-np.sum(trans_freq * np.log(trans_freq), axis=0))

    idx = np.where(maximum_filter1d(trans_ent, size=2) < threshold)[0]

    counts = count_syllables(data[:, idx])

    if pcs is not None:
        return counts, data[:, idx], pcs[idx]

    return counts, data[:, idx]


def get_topk_syllables(counts, k=2):

    top_sylls = np.argsort(counts, axis=0)[-k:]

    # zero out all counts except top k for each frame
    top_counts = np.zeros_like(counts)
    top_counts[top_sylls, np.arange(top_sylls.shape[1])] = counts[
        top_sylls, np.arange(top_sylls.shape[1])
    ]
    # normalize
    top_counts = top_counts / np.sum(top_counts, axis=0, keepdims=True)

    return top_sylls, top_counts


def compute_neighbors(
    top_sylls, top_counts, k: int = 2, ent_thresh: float = 0.5, return_mtx: bool = False
):
    """
    Args:
        data: matrix of resampled syllable labels of shape (n_resamples, n_frames)
        k: number of neighbors to save
        ent_thresh: threshold for considering high entropy frames (default: 0.5 - switching occurs more than 10% of the time)
        return_mtx: whether to return co-occurrence (i.e., confusion) matrix
    """
    k_neighbors = {}

    total_counts = np.sum(top_counts, axis=1)
    co_occurrence_mtx = np.zeros((100, 100))

    for syll in filter(lambda x: total_counts[x] > 0, range(100)):
        # only consider frames where syll is present
        mask = np.any(top_sylls == syll, axis=0)
        masked_counts = top_counts[:, mask]

        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(masked_counts * np.log2(masked_counts), axis=0)
        ent = np.nan_to_num(ent)

        masked_sylls = top_sylls[:, mask].T
        masked_sylls = masked_sylls[masked_sylls != syll]

        big_arr = np.full((100, len(ent)), np.nan, dtype="float32")
        big_arr[masked_sylls, np.arange(len(ent))] = ent

        # compute average entropy
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            avg_ent = np.nanmean(big_arr, axis=1)
        avg_ent = np.nan_to_num(avg_ent)

        # set ent to 0 if used less than current syllable
        avg_ent[total_counts < total_counts[syll]] = 0

        co_occurrence_mtx[syll] = avg_ent

        if np.any(avg_ent > 0):
            tmp_idx = np.argsort(avg_ent)[::-1]
            tmp_idx = tmp_idx[avg_ent[tmp_idx] > ent_thresh]

            if len(tmp_idx) > 0:
                k_neighbors[int(syll)] = tmp_idx[:k]

    if return_mtx:
        return k_neighbors, co_occurrence_mtx

    return k_neighbors


def to_csv(k_neighbor_map, output_path: str):
    out = Path(output_path) / "merge_candidates.csv"
    logging.info(f"Saving merge candidates to CSV: {out}")
    with open(out, "w") as f:
        f.write("labels (original) - old,labels (original) - new\n")
        for k, v in k_neighbor_map.items():
            f.write(f"{k},{v}\n")
    return out


def compute_merge_syll_stats(df, merge_map):

    vcs = df.query("onset")["labels (original)"].value_counts(normalize=True)

    for from_syll, to_syll in merge_map.items():
        logging.info(f"Syllable {from_syll} (usage {vcs[from_syll]:0.4f}) merging into {to_syll} (usage {vcs[to_syll]:0.4f})")


def merge_large_dataframe(dataframe_path, merge_map):
    df = pd.read_csv(dataframe_path, index_col=0)

    compute_merge_syll_stats(df, merge_map)

    label_map = (
        df[["labels (original)", "labels (usage sort)"]]
        .drop_duplicates()
        .set_index("labels (original)")["labels (usage sort)"]
    )
    logging.info("Mapping syllable labels")

    df["new labels (original)"] = df["labels (original)"].replace(merge_map)
    df["new labels (usage sort)"] = df["new labels (original)"].map(label_map)

    logging.info(f"Percent similar after merge: {(df['new labels (original)'] == df['labels (original)']).mean() * 100}")
    logging.info(f"Percent similar after merge (sorted): {(df['new labels (usage sort)'] == df['labels (usage sort)']).mean() * 100}")

    return df


def make_resample_plots(data, merge_map, output_path, max_width=3000):
    import matplotlib.pyplot as plt

    folder = Path(output_path) / "resample_plots"
    folder.mkdir(parents=True, exist_ok=True)

    for k, v in merge_map.items():
        mask = np.any(data == k, axis=0)
        to_plt = data[:, mask]
        if to_plt.shape[1] > max_width:
            mask = np.where(np.any(to_plt == v, axis=0))[0]
            idx = np.concatenate([np.arange(max_width - len(mask)), mask])
            to_plt = to_plt[:, idx]
        to_plt = (to_plt == k) * 1 + (to_plt == v) * 2
        fig = plt.figure()
        plt.imshow(to_plt, aspect='auto', cmap="cubehelix")
        cb = plt.colorbar(label="Syllable")
        cb.set_ticks([0, 1, 2])
        cb.set_ticklabels(["other", k, v])
        plt.title(f"Syllable {k} merging into {v}")
        fig.tight_layout()
        fig.savefig(folder / f"syllable_{k}_merging_{v}.png", dpi=300)
        plt.close(fig)
        


def print_and_log(statement):
    print(statement)
    logging.info(statement)


@click.command()
@click.argument("model_resample_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("pc_scores_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--max-syllable-labels",
    type=int,
    default=100,
    help="Maximum number of syllable labels MoSeq model was trained on",
)
@click.option(
    "--ent-thresh",
    type=float,
    default=0.5,
    help="Threshold for determining merge candidates. 0.5 ~ 10% of frames switch between syllables",
)
@click.option(
    "--output-path",
    default=Path.cwd(),
    type=click.Path(dir_okay=True, exists=False),
    help="Folder to save output",
)
@click.option(
    "--merge-mode",
    type=click.Choice(["info", "auto"]),
    default="info",
    help="Mode for determining merge candidates. 'info' only shows information, 'auto' performs merging",
)
@click.option(
    "--dataframe-path",
    default=None,
    type=click.Path(dir_okay=False, exists=True),
    help="Path to moseq_df.csv dataframe of syllable labels. Required if merge-mode is 'auto'",
)
def main(
    model_resample_path,
    pc_scores_path,
    max_syllable_labels,
    ent_thresh,
    output_path,
    merge_mode,
    dataframe_path,
):
    """
    Args:
        model_resample_path: Path to MoSeq model file - must contain resampled syllable labels (--save-every should be > 0)
    """
    log_path = Path(output_path) / "merge_syllables.log"
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode="w")
    logging.info(f"Running merge-syllables.py with merge-mode: {merge_mode}")

    uuids, data = load_syllables(model_resample_path)
    pcs = load_pcs(pc_scores_path, uuids)

    if pcs.shape[0] != data.shape[1]:
        raise ValueError(
            f"Number of frames in pcs and syllables do not match: {pcs.shape[0]} != {data.shape[1]}"
        )

    logging.info("Filtering frames to disregard transitions")
    counts, filtered_data, filtered_pcs = get_low_entropy_frame_counts(data, pcs)

    logging.info("Getting top 2 syllables for each frame")
    top_sylls, top_counts = get_topk_syllables(counts, k=2)

    logging.info("Computing syllable similarity via PCs")
    # use MAP estimate of syllable sequence
    pc_similarity = compute_pc_similarity(
        filtered_pcs, top_sylls[-1], max_label=max_syllable_labels
    )

    logging.info("Getting first draft of merge candidates")
    # compute neighbors
    k_neighbors, co_occurrence_mtx = compute_neighbors(
        top_sylls, top_counts, k=2, return_mtx=True, ent_thresh=ent_thresh
    )
    logging.info(f"Found {len(k_neighbors)} merge candidates")

    def is_similar(in_syll, out_syll):
        return pc_similarity[in_syll, out_syll] > 0

    logging.info(
        "Filtering merge candidates based on PC similarity - must be kinematically similar"
    )
    k_neighbors = {
        k: list(filter(partial(is_similar, k), v)) for k, v in k_neighbors.items()
    }
    k_neighbors = valfilter(len, k_neighbors)

    merge_map = {int(k): int(v[0]) for k, v in k_neighbors.items()}
    logging.info(f"Final set of merge candidates has {len(merge_map)} candidates")

    for from_syll, to_syll in merge_map.items():
        logging.info(f"Syllable {from_syll} (pc similarity {pc_similarity[from_syll, to_syll]:0.4f}) merging into {to_syll}")

    save_path = to_csv(merge_map, output_path)
    print_and_log(f"Saved merge candidates to {save_path}")

    logging.info("Making resample plots")
    make_resample_plots(filtered_data, merge_map, output_path)

    # step 1: re-write main dataframe
    if merge_mode == "auto":
        folder = Path(dataframe_path).parent
        print_and_log("Merging syllables in dataframe")

        df = merge_large_dataframe(dataframe_path, merge_map)
        df.to_csv(folder / "moseq_df_merged.csv")

        logging.info("Recomputing behavioral statistics dataframes")
        stats_df = compute_behavioral_statistics(
            df,
            count="usage",
            syllable_key="new labels (usage sort)",
            usage_normalization=True,
            groupby=["group", "uuid"],
        )

        stats_df_orig_labels = compute_behavioral_statistics(
            df,
            count="usage",
            syllable_key="new labels (original)",
            usage_normalization=True,
            groupby=["group", "uuid"],
        )

        stats_df.to_csv(folder / "moseq_df_stats_merged.csv")
        stats_df_orig_labels.to_csv(folder / "moseq_df_orig_label_stats_merged.csv")
        print_and_log("Saved behavioral statistics dataframes")
    
    print_and_log("Finished")


if __name__ == "__main__":
    main()
