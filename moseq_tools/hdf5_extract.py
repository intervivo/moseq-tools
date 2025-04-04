import argparse
import os
import sys
import json
import csv
import h5py
import numpy as np

def list_datasets(hdf_file):
    """List all datasets in the HDF5 file with type and shape."""
    datasets = []

    def visitor(name, node):
        if isinstance(node, h5py.Dataset):
            dtype = node.dtype
            shape = node.shape
            datasets.append((name, shape, str(dtype)))

    hdf_file.visititems(visitor)
    return datasets

def list_scalars(hdf_file):
    """List all scalars in the HDF5 file with type and shape."""
    scalars = []

    def visitor(name, node):
        if isinstance(node, h5py.Dataset):
            dtype = node.dtype
            shape = node.shape
            if name.startswith('scalars/'):
                scalars.append((name, shape, str(dtype)))

    hdf_file.visititems(visitor)
    return scalars



def export_dataset(data, output_path):
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    try:
        if np.isscalar(data):
            # Export scalar
            if ext == ".json":
                with open(output_path, "w") as f:
                    json.dump({"value": data}, f)
            elif ext == ".csv":
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["value"])
                    writer.writerow([data])
            else:
                raise ValueError("Unsupported export format. Use .csv or .json")
        else:
            # Export array
            if ext == ".json":
                with open(output_path, "w") as f:
                    json.dump(data.tolist(), f)
            elif ext == ".csv":
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    if data.ndim == 1:
                        for row in data:
                            writer.writerow([row])
                    elif data.ndim == 2:
                        for row in data:
                            writer.writerow(row)
                    else:
                        raise ValueError("Only 1D or 2D arrays can be exported to CSV.")
            else:
                raise ValueError("Unsupported export format. Use .csv or .json")
        print(f"Dataset exported to: {output_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

def export_with_context(data, timestamps, dataset_name, output_path):
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    try:
        if ext == ".json":
            output = {
                "data": data.tolist() if not np.isscalar(data) else data,
                "timestamps": timestamps.tolist() if timestamps is not None else None
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

        elif ext == ".csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["timestamp", dataset_name]
                writer.writerow(header)

                values = np.atleast_1d(data)
                n = len(values)

                for i in range(n):
                    ts = timestamps[i] if timestamps is not None and i < len(timestamps) else ""
                    val = values[i]
                    writer.writerow([ts, val])
        else:
            raise ValueError("Unsupported export format. Use .csv or .json")
        print(f"📁 Dataset exported with frames and timestamps to: {output_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

def extract_dataset(file_path, dataset_name, output_path=None):
    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    with h5py.File(file_path, 'r') as f:
        if dataset_name:
            dataset_path = f"scalars/{dataset_name}"
            if dataset_path not in f:
                print(f"❌ Dataset '{dataset_name}' not found in file.")
                sys.exit(1)

            data = f[dataset_path][()]
            if np.isscalar(data):
                print(f"✅ Scalar value from '{dataset_name}': {data}")
            else:
                print(f"✅ Array from '{dataset_name}' (shape {data.shape}):\n{data}")

            # Load context arrays (if present)
            timestamps = f["/timestamps"][()] if "/timestamps" in f else None
            if timestamps is not None:
                print(f"Found /timestamps (length={len(timestamps)})")
            if output_path:
                export_with_context(data, timestamps, dataset_name, output_path)

        else:
            datasets = list_scalars(f)
            if datasets:
                print("Available scalars:")
                for name, shape, dtype in datasets:
                    kind = "Scalar" if shape == () else "Array"
                    print(f"  - {name.replace('scalars/', '')} [{kind}] Shape: {shape} Type: {dtype}")
            else:
                print("No datasets found.")

def main():
    parser = argparse.ArgumentParser(description="Extract an array from an HDF5 file.")
    parser.add_argument("file", help="Path to the HDF5 file")
    parser.add_argument("-d", "--dataset", help="Path to the dataset to extract", default=None)
    parser.add_argument("-o", "--output", help="Export result to a .csv or .json file", default=None)

    args = parser.parse_args()
    extract_dataset(args.file, args.dataset, args.output)

if __name__ == "__main__":
    main()
