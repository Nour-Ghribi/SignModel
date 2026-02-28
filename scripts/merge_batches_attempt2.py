import argparse
import glob
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge extracted batch NPZ files.")
    parser.add_argument("--batches-dir", required=True, help="Directory containing batch_*_landmarks.npz")
    parser.add_argument("--out-file", required=True, help="Merged .npz output file")
    return parser.parse_args()


def main():
    args = parse_args()
    batches_dir = Path(args.batches_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(glob.glob(str(batches_dir / "batch_*_landmarks.npz")))
    if not batch_files:
        raise SystemExit("No batch landmark files found.")

    xs = []
    ys = []
    for batch_file in batch_files:
        data = np.load(batch_file)
        xs.append(data["X"])
        ys.append(data["y"])
        print(f"Loaded {batch_file}: X={data['X'].shape} y={data['y'].shape}")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.savez_compressed(out_file, X=X, y=y)

    labels_path = batches_dir / "labels.json"
    if labels_path.exists():
        merged_labels_path = out_file.with_name(out_file.stem + "_labels.json")
        with labels_path.open("r", encoding="utf-8") as src, merged_labels_path.open("w", encoding="utf-8") as dst:
            json.dump(json.load(src), dst, ensure_ascii=True, indent=2)

    print(f"Merged dataset saved to: {out_file}")
    print(f"Final shape X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
