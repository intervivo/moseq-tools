import sys
import shutil
from pathlib import Path
from ruamel.yaml import YAML

def get_file_index(file_name, yaml_data):
    """
    Get the index of a file in the YAML data based on its name.
    """
    for i, file in enumerate(yaml_data['files']):
        if file['path'][0].endswith(file_name):
            return i
    return None

def main(from_index, to_index):
    yaml_reader = YAML(typ='safe')

    with open(from_index, 'r') as f:
        from_yaml = yaml_reader.load(f)

    with open(to_index, 'r') as f:
        to_yaml = yaml_reader.load(f)

    if not Path(to_index + ".bak").exists():
        shutil.copyfile(to_index, to_index + ".bak")

    # Merge the groups of the two YAML files

    for i in range(len(to_yaml['files'])):
        # find the associated file in the from_yaml
        file_name = Path(to_yaml['files'][i]['path'][0]).name
        file_idx = get_file_index(file_name, from_yaml)
        if file_idx is not None:
            # If the file exists in both, merge the groups
            from_group = from_yaml['files'][file_idx]['group']
            to_group = to_yaml['files'][i]['group']
            if from_group != to_group:
                print(f"Setting group for {file_name} from {to_group} to {from_group}")
                to_yaml['files'][i]['group'] = from_group
        else:
            print(f"File {file_name} not found in the source YAML.")

    with open(to_index, 'w') as f:
        yaml_reader.dump(to_yaml, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_yaml.py <from_index> <to_index>")
        sys.exit(1)

    from_index = sys.argv[1]
    to_index = sys.argv[2]

    main(from_index, to_index)