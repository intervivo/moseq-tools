#!/usr/bin/env python
import h5py
import click
import shutil
import joblib
import logging
import subprocess
import numpy as np
from pathlib import Path
from pprint import pformat


def flip_frames(files, flip_qc_folder):
    for file in files:
        logging.info(f"Copying {file} to {flip_qc_folder}")
        new_file_path = Path(flip_qc_folder) / file.name
        if new_file_path.exists():
            logging.warning(
                f"{new_file_path} already exists, assuming flipped and skipping"
            )
            continue

        shutil.copy2(file, new_file_path)
        # copy yaml file too
        shutil.copy2(file.with_suffix(".yaml"), new_file_path.with_suffix(".yaml"))

        with h5py.File(new_file_path, "r+") as h5f:
            # flip frames
            frames = h5f["frames"][()]
            flipped_frames = np.flip(frames, axis=1)
            h5f["frames"][:] = flipped_frames

        logging.info(f"Flipped frames in {new_file_path}")


def compare_syllable_similarity(original_model_file, flipped_model_file):
    
    mdl1 = joblib.load(original_model_file)
    mdl2 = joblib.load(flipped_model_file)

    mdl1_labels = dict(zip(mdl1['keys'], mdl1['labels']))
    mdl2_labels = dict(zip(mdl2['keys'], mdl2['labels']))

    # 1: compute overall similarity
    overall_similarity = {k: (mdl1_labels[k] == mdl2_labels[k]).mean() for k in mdl1_labels}
    avg_similarity = np.mean(list(overall_similarity.values())) * 100
    logging.info(f"% of matching syllable IDs: {avg_similarity:.2f}")

    # 2: compute per-syllable similarity
    per_syllable_similarity = {}
    for syll in range(100):
        samples = []
        for k, v in mdl1_labels.items():
            mask = v == syll
            if np.any(mask):
                samples.append((v[mask] == mdl2_labels[k][mask]).mean())
            else:
                samples.append(np.nan)
        per_syllable_similarity[syll] = np.nanmean(samples) * 100

    per_syllable_similarity = dict(sorted(per_syllable_similarity.items(), key=lambda item: item[1], reverse=True))
    logging.info("Per-syllable similarity:")
    logging.info(pformat(per_syllable_similarity, sort_dicts=False))

    return per_syllable_similarity, avg_similarity


@click.command()
@click.argument("aggregate_folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--flip-qc-folder",
    default="flip_qc",
    type=click.Path(file_okay=False),
    help="Folder to save flip qc intermediate files",
)
@click.option(
    "--pca-folder",
    default="_pca",
    type=click.Path(file_okay=False),
    help="Folder containing PCA results",
)
@click.option(
    "--model-file",
    type=click.Path(dir_okay=False, exists=True),
    help="Path to trained MoSeq model file",
)
def main(aggregate_folder, flip_qc_folder, pca_folder, model_file):
    """
    Runs a quality control check to assess syllable similarity after flipping frames.

    Args:
        aggregate_folder: Path to folder containing aggregated extraction h5 files
        flip_qc_folder: Folder to save flip qc intermediate files
        pca_folder: Folder containing PCA results
        model_file: Path to trained MoSeq model file
    """
    log_path = Path(flip_qc_folder) / "frame_flip_qc.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode="w")

    files = sorted(Path(aggregate_folder).glob("*.h5"))

    # step 1: flip frames U-D
    logging.info("Flipping frames up-down")
    flip_frames(files, flip_qc_folder)

    flip_model_file = Path(model_file).parent / "flipped_frames_model_results.p"

    if not flip_model_file.exists():
        # step 2: apply PCA
        logging.info("Applying PCA")
        subprocess.run(
            [
                "moseq2-pca",
                "apply-pca",
                "-i",
                flip_qc_folder,
                "-o",
                pca_folder,
                "--output-file",
                "flipped_pca_scores",
                "--cluster-type",
                "local",
                "--overwrite-pca-apply",
                "1",
                "--config-file",
                "config.yaml",
            ],
            check=True,
        )

        # step 3: apply moseq model
        logging.info("Applying MoSeq model")
        subprocess.run(
            [
                "moseq2-model",
                "apply-model",
                model_file,
                str(Path(pca_folder, "flipped_pca_scores.h5")),
                str(flip_model_file),
            ],
            check=True,
        )

    # step 4: compare syllable similarity
    results, avg_similarity = compare_syllable_similarity(
        model_file,
        str(flip_model_file),
    )
    print(f"% of matching syllable IDs: {avg_similarity:.0f}%")

    print("Per-syllable matching %:")
    print(pformat(results, sort_dicts=False))

    # step 5: save results
    with open(Path(flip_qc_folder) / "frame_flip_qc_results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.0f}%\n")
    

if __name__ == "__main__":
    main()
