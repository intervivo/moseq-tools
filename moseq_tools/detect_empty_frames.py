import h5py
import click
import shutil
import logging
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from scipy.ndimage import maximum_filter1d
from toolz import concat, partition
from ruamel.yaml import YAML


def detect_empty_frame_ranges(
    file, mouse_height_threshold=10, split_threshold=150, filter_size=9
):
    """
    Detect ranges of frames in a file that are empty (i.e. contain only zeros).

    Parameters
    ----------
    file : str
        Path to the HDF5 file to be processed.
    mouse_height_threshold : int, optional
        All frames with a height value below this threshold are considered empty.
        Default is 10.
    split_threshold : int, optional
        Minimum number of frames required to trigger a split.
        Default is 150.
    filter_size : int, optional
        Size of the maximum filter used to smooth the empty frame mask.
        Default is 9.

    Returns
    -------
    A list of tuples (start, end) where start and end are frame indices.
    """

    with h5py.File(file, "r") as f:
        frames = f["frames"][()]
        mouse_mask = np.sum(frames > mouse_height_threshold, axis=(1, 2)) == 0
        # write a min filter of 5 frames wide
        filtered_mask = maximum_filter1d(mouse_mask, size=filter_size, mode="nearest")

        start = np.where(np.diff(filtered_mask.astype("int8")) == 1)[0]
        end = np.where(np.diff(filtered_mask.astype("int8")) == -1)[0]

        if len(start) - 1 == len(end):
            end = np.append(end, len(mouse_mask))
        last_index = end[-1]
    durations = np.array([e - s for s, e in zip(start, end)])


    logging.info("Onsets " + str(start.tolist()))
    logging.info("Offsets " + str(end.tolist()))
    logging.info("Durations " + str(durations.tolist()))

    problem_frames = durations > split_threshold
    places_to_split = np.where(problem_frames)[0]
    start, end = start[places_to_split], end[places_to_split]
    splits = list(zip(start, end))

    if not splits:
        return None

    # if time between last end and next start < 2 * split_threshold frames, merge the splits
    new_splits = [splits[0]]
    for s, e in splits[1:]:
        cur_split = new_splits[-1]
        # merge empty frame windows
        if s - cur_split[-1] < (split_threshold * 2):
            new_splits[-1] = (cur_split[0], e)
        else:
            new_splits.append((s, e))
    
    logging.info("Old splits " + str(splits))
    logging.info("New merged splits " + str(new_splits))

    # new splits contain empty frames - invert to range of non-empty frames
    ranges = concat([(0,)] + new_splits + [(len(mouse_mask),)])
    inverse_splits = list(partition(2, ranges))
    if inverse_splits[-1][0] == len(mouse_mask):
        inverse_splits.pop()

    if last_index != len(mouse_mask) and inverse_splits[-1][1] != len(mouse_mask):
        inverse_splits.append((new_splits[-1][1], len(mouse_mask)))

    return inverse_splits


def split_h5(file, new_file_name, start, end, index):
    with h5py.File(file, "r") as f, h5py.File(new_file_name, "w") as new_f:

        def _recursive_copy(src_obj, dest_obj, start=start, end=end):
            for k, v in src_obj.items():
                if isinstance(v, h5py.Dataset):
                    # Handle different dataset types
                    if v.shape == () or isinstance(v[()], (str, np.str_)):
                        # Scalar or string dataset
                        if k == "uuid":
                            try:
                                dest_obj.create_dataset(
                                    k,
                                    data=v[()].decode() + f"-split-{index:02d}",
                                )
                            except Exception:
                                dest_obj.create_dataset(
                                    k,
                                    data=v[()] + f"-split-{index:02d}",
                                )

                        else:
                            dest_obj.create_dataset(k, data=v[()])
                    else:
                        try:
                            # Numeric or array dataset
                            dest_obj.create_dataset(
                                k, data=v[start:end], compression="gzip"
                            )
                        except ValueError:
                            # Fallback for complex objects
                            try:
                                # Try to copy the entire dataset if slicing fails
                                dest_obj.create_dataset(k, data=v[()], dtype=v.dtype)
                                logging.warning(
                                    f"Could not slice dataset {k}. Copied entire dataset."
                                )
                            except Exception as copy_err:
                                logging.error(f"Error copying dataset {k}: {copy_err}")

                elif isinstance(v, h5py.Group):
                    # Recursively handle groups
                    new_group = dest_obj.create_group(k)
                    _recursive_copy(v, new_group)

        # Start the recursive copy
        _recursive_copy(f, new_f)


@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.option(
    "--file-mode",
    default=None,
    type=click.Choice(["i", "a", "n"]),
    help="Mode for handling empty frames. (i)nteractive, (a)utomatic, (n)othing. Default: nothing",
)
@click.option(
    "--empty-threshold",
    default=10,
    type=float,
    help="Percentage of empty frames that triggers a warning.",
)
@click.option(
    "--split-threshold",
    default=150,
    type=int,
    help="Minimum number of frames to consider splitting a file.",
)
@click.option(
    "--mouse-height-threshold",
    default=10,
    type=int,
    help="Anything below this threshold (in millimeters) is considered empty/noise.",
)
@click.option(
    "--print-indices", is_flag=True, help="Print the indices of the empty frames."
)
def main(
    input_folder,
    file_mode,
    empty_threshold,
    split_threshold,
    mouse_height_threshold,
    print_indices,
):
    """
    Detect empty frames in files in a folder and split the files based on these.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the HDF5 files to be processed.
    file_mode : str, optional
        Mode for handling empty frames. (i)nteractive, (a)utomatic, (n)othing. Default: nothing
    empty_threshold : float, optional
        Fraction of empty frames that triggers a warning. Default is 0.1.
    split_threshold : int, optional
        Minimum number of frames to consider splitting a file. Default is 150.
    mouse_height_threshold : int, optional
        Anything below this threshold (in millimeters) is considered empty/noise. Default is 10.
    print_indices : bool, optional
        Print the indices of the empty frames. Default is False.
    """
    yaml_loader = YAML(typ="safe", pure=True)

    # write log to file
    log_file = Path(input_folder) / "detect-empty-frames.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w")
    logging.info(
        f"Starting detection of empty frames in {input_folder}. File mode: {file_mode}"
    )

    files = sorted(Path(input_folder).glob("*.h5"))
    empty_files = []
    for file in tqdm(files):
        logging.info(f"Processing {file}")
        with h5py.File(file, "r") as f:
            frames = f["frames"][()]

            mouse_mask = np.sum(frames > 10, axis=(1, 2)) == 0

            empty_pct = np.mean(mouse_mask) * 100
            empty_total = np.sum(mouse_mask)

            if empty_pct >= empty_threshold:
                empty_files.append(file)
                print(
                    f"\nFile {file} has {empty_pct:0.0f}% empty frames: {empty_total}."
                )
                logging.info(
                    f"Warning: file {file} has {empty_pct:0.0f}% empty frames: {empty_total}."
                )
                if file_mode == "a":
                    frame_ranges = detect_empty_frame_ranges(
                        file,
                        mouse_height_threshold=mouse_height_threshold,
                        split_threshold=split_threshold,
                    )
                    if not frame_ranges:
                        logging.info("File passed test. Skipping.")
                        continue
                    if print_indices:
                        print("Split file frame ranges", frame_ranges)
                    for i, (s, e) in enumerate(tqdm(frame_ranges, leave=False, desc="Splitting file...")):
                        new_file_name = file.with_name(
                            file.stem + f"-start_{s}_end_{e}.h5"
                        )
                        logging.info(
                            f"Splitting {file} with range {s} - {e} into {new_file_name}"
                        )
                        split_h5(file, new_file_name, s, e, i)
                        # if yaml file exists, make a copy with same name as h5 file
                        yaml_file = file.with_suffix(".yaml")
                        if yaml_file.exists():
                            new_yaml = new_file_name.with_suffix(".yaml")
                            shutil.copy2(yaml_file, new_yaml)
                            with open(new_yaml, "r") as f:
                                yaml_content = yaml_loader.load(f)
                                yaml_content["uuid"] += f"-split-{i:02d}"
                            with open(new_yaml, "w") as f:
                                yaml_loader.dump(yaml_content, f)

                    # move original file to backup
                    file.rename(file.with_name(file.name + ".orig"))

            logging.info(f"Fraction of empty frames: {empty_pct:0.0f}%")
            logging.info(f"Total number of empty frames: {empty_total} frames")

    if file_mode == "i":
        print("List of empty files:", empty_files)
        input("Remove or rename files. Once done, press enter...")

    logging.info("Done")


if __name__ == "__main__":
    main()
