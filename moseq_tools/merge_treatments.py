import click
import shutil
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML


@click.command()
@click.argument('table_file_map', type=click.Path(exists=True)) 
@click.argument('table_dosing_schedule', type=click.Path(exists=True))
@click.argument('moseq_index_path', type=click.Path(exists=True))
def main(table_file_map, table_dosing_schedule, moseq_index_path):
    """Merge treatment information from two tables and update the MoSeq index file.
    Args:
        table_file_map (str): Path to the file map table (tab-separated).
        table_dosing_schedule (str): Path to the dosing schedule table (tab-separated).
        moseq_index_path (str): Path to the MoSeq index file (YAML).
    """
    yaml = YAML(typ='safe')
    df = pd.read_csv(table_file_map, sep="\t")
    df2 = pd.read_csv(table_dosing_schedule, sep="\t")

    df = df.merge(df2, how="outer", on=["matcher", "day", "session", "box"])

    df['filename_clean'] = df['filename'].str.replace('.mp4', '')

    filename_map = df.set_index('filename_clean')['treat'].to_dict()

    if not Path(moseq_index_path + ".bak").exists():
        shutil.copyfile(moseq_index_path, moseq_index_path + ".bak")

    with open(moseq_index_path, 'r') as f:
        moseq_index = yaml.load(f)

    for i in range(len(moseq_index['files'])):
        file = moseq_index['files'][i]
        filename = Path(file['path'][0]).stem

        new_grp = filename_map.get(filename, file['group'])
        if new_grp == file['group']:
            print(f"File {filename} already had group {new_grp}. New group not found.")

        moseq_index['files'][i]['group'] = new_grp

    with open(moseq_index_path, 'w') as f:
        yaml.dump(moseq_index, f)


if __name__ == "__main__":
    main()