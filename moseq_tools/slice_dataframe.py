import click
import pandas as pd
from pathlib import Path


# hard-coded default values
FILE_COPY_SUFFIX = "_copy"


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--time-slice",
    "-s",
    type=(float, float),
    default=(0, 30),
    help="Time slice (in minutes) to extract. Can be float. Ex: --time-slice 0.0 30",
)
@click.option(
    "--output_file",
    "-o",
    default=None,
    help="Output file name (optional). Ex: -o moseq_slice_0_30.csv",
)
def slice_dataframe(input_file, time_slice, output_file):
    """
    Filter a CSV dataframe to only include data within a specified time range.

    INPUT_FILE: Path to the input CSV file
    """

    input_file = Path(input_file)

    df = pd.read_csv(input_file)

    df = df.filter(regex="^(?!Unnamed)")  # remove unnamed columns

    # convert time slice to milliseconds
    frame_slice = tuple(int(time * 60 * 1000) for time in time_slice)

    df["aligned_timestamps"] = df.groupby(["SessionName", "SubjectName", "uuid"])[
        "timestamps"
    ].transform(lambda x: x - x.min())
    df = df[df["aligned_timestamps"].between(*frame_slice)]
    df = df.drop("aligned_timestamps", axis=1)

    if output_file is None:
        output_path = input_file.with_name(input_file.stem + FILE_COPY_SUFFIX + ".csv")
    else:
        output_path = input_file.with_stem(Path(output_file).stem)

    df.to_csv(output_path, index=False)

    click.echo(f"Dataframe saved to {output_path}")
    click.echo(f"Time slice: {time_slice[0]} minutes to {time_slice[1]} minutes")


if __name__ == "__main__":
    slice_dataframe()
